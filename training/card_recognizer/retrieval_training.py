"""Data isolation, whole-card augmentation, and trainable retrieval components."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.nn import functional as F

from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input


def identity_key(identity: dict) -> str:
    return ':'.join(identity[k] for k in ('game', 'language', 'card_id'))


def holdout_family(key: str) -> tuple[str, str]:
    game, language, card_id = key.split(':', 2)
    # One Piece base numbers recur across languages and art suffixes.
    return (game, card_id.split('_p')[0]) if game == 'onepiece' else (game + ':' + language, card_id)


def build_training_plan(manifest: dict, max_classes: int = 2048, seed: int = 42,
                        excluded_identities: tuple[str, ...] = ()) -> dict:
    """Exclude evaluation identities/families/byte aliases before sampling references."""
    if max_classes < 2:
        raise ValueError('At least two training classes are required')
    gallery = manifest['gallery']
    queries = {s: manifest['splits'][s]['queries'] for s in ('train', 'val', 'test')}
    sources: dict[str, str] = {}
    query_hashes: dict[str, str] = {}
    for split, rows in queries.items():
        for row in rows:
            if sources.setdefault(row['source_group'], split) != split:
                raise ValueError('Source video crosses splits')
            if query_hashes.setdefault(row['sha256'], split) != split:
                raise ValueError('Query bytes cross splits')
    held = {identity_key(r['identity']) for s in ('val', 'test') for r in queries[s]}
    families = {holdout_family(k) for k in held}
    excluded = {identity_key(r['identity']) for r in gallery if holdout_family(identity_key(r['identity'])) in families}
    excluded.update(excluded_identities)
    # Close over exact-byte aliases, including transitive identities/templates.
    while True:
        hashes = {r['sha256'] for r in gallery if identity_key(r['identity']) in excluded}
        expanded = excluded | {identity_key(r['identity']) for r in gallery if r['sha256'] in hashes}
        if expanded == excluded:
            break
        excluded = expanded
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for row in gallery:
        key = identity_key(row['identity'])
        if key not in excluded:
            by_identity[key].append(row)
    # One training proxy per connected exact-image visual family, preserving IDs.
    parent = {k: k for k in by_identity}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owner = {}
    for key, rows in by_identity.items():
        for row in rows:
            previous = owner.setdefault(row['sha256'], key)
            a, b = find(key), find(previous)
            parent[max(a, b)] = min(a, b)
    groups: dict[str, dict] = {}
    for key in sorted(by_identity):
        group = groups.setdefault(find(key), {'identities': [], 'references': [], 'queries': []})
        group['identities'].append(key)
        group['references'].extend(by_identity[key])
    excluded_train = []
    for row in queries['train']:
        key = identity_key(row['identity'])
        if key in excluded:
            excluded_train.append(row)
        elif key not in parent:
            raise ValueError(f'Training query has no enrolled reference: {key}')
        else:
            groups[find(key)]['queries'].append(row)
    mandatory = sorted(k for k, g in groups.items() if g['queries'])
    if len(mandatory) > max_classes:
        raise ValueError('Class cap would omit identities with real training queries')
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        if key not in mandatory:
            buckets[key.split(':')[0]].append(key)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = list(mandatory)
    # Round-robin games rather than letting the larger Pokémon corpus dominate.
    while len(selected) < min(max_classes, len(groups)):
        for game in sorted(buckets):
            if buckets[game] and len(selected) < max_classes:
                selected.append(buckets[game].pop())
        if not any(buckets.values()):
            break
    classes = [groups[k] for k in sorted(selected)]
    allowed_ids = {k for g in classes for k in g['identities']}
    allowed_hashes = {r['sha256'] for g in classes for r in g['references'] + g['queries']}
    held_hashes = {r['sha256'] for r in gallery if identity_key(r['identity']) in excluded}
    if allowed_ids & excluded or allowed_hashes & held_hashes:
        raise ValueError('Held-out reference leakage')
    # Validation has fixed distractors, including every variant of held-out OP bases.
    validation_gallery = [r for r in gallery if identity_key(r['identity']) in allowed_ids | excluded]
    return {
        'schema_version': 1, 'seed': seed, 'classes': classes,
        'training_class_count': len(classes), 'training_catalog_identity_count': len(allowed_ids),
        'training_real_query_count': sum(len(g['queries']) for g in classes),
        'held_out_query_identities': sorted(held), 'excluded_training_reference_identities': sorted(excluded),
        'known_ambiguous_training_identities': sorted(excluded_identities),
        'excluded_training_queries': excluded_train, 'validation_gallery': validation_gallery,
        'validation_queries': queries['val'], 'test_queries': queries['test'],
        'source_group_splits': sources,
        'policy': 'Split first; exclude val/test identities, OP base families across languages, and exact-byte aliases from all training gradients. Merge exact-byte aliases only for training proxies. Test queries are evaluated once after checkpoint selection.',
    }


def load_card_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.uint8)
    return canonicalize_card(rgb, width=224, height=312)


def augment_card(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild camera effects; all four card corners remain inside the output canvas."""
    image = Image.fromarray(rgb)
    for enhancer, limits in [(ImageEnhance.Brightness, (.70, 1.25)),
                             (ImageEnhance.Contrast, (.75, 1.25)),
                             (ImageEnhance.Color, (.75, 1.25))]:
        image = enhancer(image).enhance(rng.uniform(*limits))
    if rng.random() < .35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.2, .8)))
    work = np.asarray(image, dtype=np.uint8).copy()
    if rng.random() < .25:
        h, w = work.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        center = rng.uniform(0, w)
        band = np.exp(-((xx - center - .3 * yy) / rng.uniform(20, 60)) ** 2)
        work = np.clip(work + band[..., None] * rng.uniform(12, 45), 0, 255).astype(np.uint8)
    if rng.random() < .4:
        success, encoded = cv2.imencode('.jpg', work, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
        if not success:
            raise ValueError('JPEG augmentation failed')
        work = cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # RGB channel order is retained through this symmetric roundtrip.
    h, w = work.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[rng.uniform(.02, .09) * w, rng.uniform(.02, .07) * h],
                      [rng.uniform(.91, .98) * w, rng.uniform(.02, .07) * h],
                      [rng.uniform(.91, .98) * w, rng.uniform(.93, .98) * h],
                      [rng.uniform(.02, .09) * w, rng.uniform(.93, .98) * h]])
    return cv2.warpPerspective(work, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderValue=(124, 116, 104), flags=cv2.INTER_LINEAR)


class TrainingImages:
    def __init__(self, root: Path, classes: list[dict], seed: int):
        self.root, self.classes, self.seed = root, classes, seed
        self.cache: dict[str, np.ndarray] = {}
        self.real = [i for i, g in enumerate(classes) if g['queries']]
        self.digital = [i for i, g in enumerate(classes) if not g['queries']]

    def read(self, row: dict) -> np.ndarray:
        key = row['image']
        if key not in self.cache:
            path = (self.root / key).resolve(strict=True)
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError('Image path escapes dataset')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Image hash mismatch: {path}')
            self.cache[key] = load_card_rgb(path)
        return self.cache[key]

    def batch(self, step: int, identities_per_batch: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + step * 1009)
        n_real = min(len(self.real), identities_per_batch // 2)
        chosen = rng.sample(self.real, n_real)
        remaining = [i for i in self.digital if i not in chosen]
        if len(remaining) < identities_per_batch - n_real:
            remaining = [i for i in range(len(self.classes)) if i not in chosen]
        chosen += rng.sample(remaining, identities_per_batch - n_real)
        inputs, labels = [], []
        for label in chosen:
            group = self.classes[label]
            pair = [rng.choice(group['references']), rng.choice(group['queries'] or group['references'])]
            for row in pair:
                inputs.append(prepare_encoder_input(augment_card(self.read(row), rng)))
                labels.append(label)
        return torch.from_numpy(np.stack(inputs)), torch.tensor(labels, dtype=torch.long)


def prepare_fixed_position_embeddings(backbone: nn.Module, size: int = 224) -> dict:
    """Materialize the frozen DINO grid on CPU once; avoid MPS bicubic per forward."""
    embeddings = backbone.embeddings
    original = embeddings.position_embeddings
    if original.device.type != 'cpu' or size % embeddings.patch_size:
        raise ValueError('Prepare positions on CPU at a patch-aligned fixed input size')
    count = (size // embeddings.patch_size) ** 2 + 1
    with torch.no_grad():
        positions = embeddings.interpolate_pos_encoding(
            torch.empty(1, count, original.shape[-1], dtype=original.dtype), size, size).detach().clone()
    embeddings.position_embeddings = nn.Parameter(positions, requires_grad=False)
    return {'original_tokens': original.shape[1], 'fixed_tokens': count, 'input_size': size,
            'policy': 'Original pretrained CPU bicubic interpolation once, frozen thereafter; same fixed-input outputs'}


class CardEmbedding(nn.Module):
    def __init__(self, backbone: nn.Module, dimension: int = 384):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.projection = nn.Linear(hidden, dimension, bias=False)
        if dimension == hidden:
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.orthogonal_(self.projection.weight)

    def forward(self, pixels):
        if tuple(pixels.shape[-2:]) != (224, 224):
            raise ValueError('This encoder contract requires fixed 224x224 pixels')
        cls = self.backbone(pixel_values=pixels).last_hidden_state[:, 0, :]
        return F.normalize(self.projection(cls).float(), dim=1)

    def configure_backbone(self, last_blocks: int):
        layers = self.backbone.encoder.layer
        if not 0 <= last_blocks <= len(layers):
            raise ValueError('Invalid number of trainable backbone blocks')
        self.backbone.requires_grad_(False)
        for layer in list(layers)[len(layers) - last_blocks:] if last_blocks else []:
            layer.requires_grad_(True)
        if last_blocks:
            self.backbone.layernorm.requires_grad_(True)


class ArcFace(nn.Module):
    """Normalized training proxies with monotonic additive angular margin (radians)."""
    def __init__(self, dimension: int, classes: int, margin: float = .2, scale: float = 16):
        super().__init__()
        if dimension < 1 or classes < 2 or not 0 <= margin < math.pi / 2 or scale <= 0:
            raise ValueError('Invalid ArcFace dimensions/margin/scale')
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        nn.init.normal_(self.weight, std=.01)
        self.margin, self.scale = margin, scale

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features.float(), dim=1), F.normalize(self.weight.float(), dim=1))
        cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)
        target = cosine.gather(1, labels[:, None])
        phi = target * math.cos(self.margin) - (1 - target.square()).clamp_min(1e-7).sqrt() * math.sin(self.margin)
        threshold = math.cos(math.pi - self.margin)
        phi = torch.where(target > threshold, phi, target - math.sin(math.pi - self.margin) * self.margin)
        return cosine.scatter(1, labels[:, None], phi) * self.scale
