#!/usr/bin/env python3
"""Safely copy root-reviewed image records into a normalized dataset layout.

The tool never assigns identities. Recognition queries are admitted only from a
root-reviewed manifest with explicit physical-reference evidence. Detector
imports are restricted to the exact synthetic renderer output contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
SPLITS = {'train', 'val', 'test'}
COMPONENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


class CurationError(ValueError):
    """Input violates the curation contract or a filesystem safety check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, stop: Path) -> None:
    """Reject a symlink at the leaf or along the path between path and stop."""
    path_abs, stop_abs = path.absolute(), stop.absolute()
    if not _under(path_abs, stop_abs):
        raise CurationError(f'Path is outside allowed root: {path}')
    cursor = stop_abs
    try:
        rel = path_abs.relative_to(stop_abs)
    except ValueError as exc:
        raise CurationError(f'Path is outside allowed root: {path}') from exc
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CurationError(f'Symlinks are not allowed in curated paths: {cursor}')


def resolve_input_file(value: str, root: Path, *, image: bool = False) -> Path:
    rel = Path(value)
    if rel.is_absolute() or not rel.parts or any(part in {'.', '..'} for part in rel.parts):
        raise CurationError(f'Input paths must be clean relative paths: {value!r}')
    lexical = root / rel
    _reject_symlink_components(lexical, root)
    path = lexical.resolve(strict=True)
    if not _under(path, root.resolve()):
        raise CurationError(f'Input path escapes its root: {value!r}')
    if not path.is_file():
        raise CurationError(f'Input is not a regular file: {path}')
    if image:
        verify_image(path)
    return path


def verify_image(path: Path) -> tuple[int, int, str]:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise CurationError(f'Unsupported image type: {path}')
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = image.format or ''
    except (OSError, UnidentifiedImageError) as exc:
        raise CurationError(f'Invalid or undecodable image: {path}') from exc
    if width < 1 or height < 1:
        raise CurationError(f'Image has invalid dimensions: {path}')
    return width, height, image_format


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurationError(f'Cannot read JSON manifest: {path}') from exc
    if not isinstance(value, dict):
        raise CurationError(f'Manifest must contain a JSON object: {path}')
    return value


def _load_manifest(path: Path, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _json_file(path)
    if payload.get('schema_version') != 1:
        raise CurationError(f'{path} must declare schema_version: 1')
    rows = payload.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CurationError(f'{path} must contain a {key} array of objects')
    return payload, rows


def _component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not COMPONENT_RE.fullmatch(value):
        raise CurationError(f'{field} must be a safe, nonempty path component; got {value!r}')
    return value


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    identity = row.get('identity')
    if not isinstance(identity, dict):
        raise CurationError('Accepted query needs an explicit identity object')
    return (
        _component(identity.get('game'), 'identity.game'),
        _component(identity.get('language'), 'identity.language'),
        _component(identity.get('card_id'), 'identity.card_id'),
    )


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CurationError(f'{field} must be a lowercase SHA-256 hex digest')
    return value


def _copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise CurationError(f'Refusing to overwrite output: {destination}')
    if mode == 'hardlink':
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _verify_sha(path: Path, claimed: Any, field: str) -> str:
    expected = _require_sha(claimed, field)
    actual = sha256_file(path)
    if actual != expected:
        raise CurationError(f'{field} mismatch for {path}: expected {expected}, found {actual}')
    return actual


def _load_references(manifest_path: Path, input_root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]], set[str]]:
    payload, references = _load_manifest(manifest_path, 'references')
    del payload
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    hashes: set[str] = set()
    for row in references:
        game = _component(row.get('game'), 'reference.game')
        language = _component(row.get('language'), 'reference.language')
        card_id = _component(row.get('card_id'), 'reference.card_id')
        path = resolve_input_file(row.get('path', ''), input_root, image=True)
        digest = _verify_sha(path, row.get('sha256'), 'reference.sha256')
        key = (game, language, card_id, digest)
        if key in by_key:
            raise CurationError(f'Duplicate gallery reference record: {key}')
        by_key[key] = {**row, '_resolved_path': path, '_identity': (game, language, card_id), '_sha256': digest}
        hashes.add(digest)
    return list(by_key.values()), by_key, hashes


def _prepare_queries(
    review_path: Path,
    review: dict[str, Any],
    input_root: Path,
    ref_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    gallery_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if review.get('reviewer') != 'root-sol':
        raise CurationError('Root review manifest must set reviewer to "root-sol"')
    if not isinstance(review.get('review_provenance'), str) or not review['review_provenance'].strip():
        raise CurationError('Root review manifest needs nonempty review_provenance')
    payload, queries = _load_manifest(review_path, 'queries')
    rejected = payload.get('rejected', [])
    if not isinstance(rejected, list) or any(not isinstance(row, dict) for row in rejected):
        raise CurationError('Review manifest rejected must be an array of objects')

    prepared: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    query_split_by_hash: dict[str, str] = {}
    destinations: set[tuple[tuple[str, str, str], str, str]] = set()
    for row in queries:
        if row.get('status') != 'verified_reference_match':
            raise CurationError('Only status=verified_reference_match can enter recognition datasets')
        if row.get('single_complete_card') is not True:
            raise CurationError('Recognition query must explicitly set single_complete_card=true')
        identity = _identity(row)
        split = row.get('split')
        if split not in SPLITS:
            raise CurationError(f'Unsupported recognition split: {split!r}')
        source_group = row.get('source_group')
        if not isinstance(source_group, str) or not source_group.strip():
            raise CurationError('Accepted query needs a nonempty source_group')
        previous_split = groups.setdefault(source_group, split)
        if previous_split != split:
            raise CurationError(f'Source group {source_group!r} crosses {previous_split}/{split}')

        crop = resolve_input_file(row.get('crop_path', ''), input_root, image=True)
        crop_sha = _verify_sha(crop, row.get('crop_sha256'), 'crop_sha256')
        source = resolve_input_file(row.get('source_path', ''), input_root, image=True)
        source_sha = _verify_sha(source, row.get('source_sha256'), 'source_sha256')
        reference_path = resolve_input_file(row.get('reference_path', ''), input_root, image=True)
        reference_sha = _verify_sha(reference_path, row.get('reference_sha256'), 'reference_sha256')
        key = (*identity, reference_sha)
        reference = ref_by_key.get(key)
        if reference is None or reference['_resolved_path'] != reference_path:
            raise CurationError('Reference path/hash/identity does not match an enrolled reference-manifest record')
        if split in {'val', 'test'} and crop_sha in gallery_hashes:
            raise CurationError(f'Evaluation crop duplicates gallery bytes: {crop}')
        previous_query_split = query_split_by_hash.setdefault(crop_sha, split)
        if previous_query_split != split:
            raise CurationError(f'Identical query bytes cross {previous_query_split}/{split} splits')

        annotation_provenance = row.get('annotation_provenance')
        annotation_notes = row.get('annotation_notes')
        if not isinstance(annotation_provenance, str) or not annotation_provenance.strip():
            raise CurationError('Accepted query needs annotation_provenance')
        if not isinstance(annotation_notes, str) or not annotation_notes.strip():
            raise CurationError('Accepted query needs annotation_notes')
        destination_key = (identity, split, crop_sha)
        if destination_key in destinations:
            raise CurationError(f'Duplicate accepted query record: {destination_key}')
        destinations.add(destination_key)
        prepared.append({
            **row,
            '_identity': identity,
            '_split': split,
            '_crop': crop,
            '_crop_sha256': crop_sha,
            '_source': source,
            '_source_sha256': source_sha,
            '_reference': reference_path,
            '_reference_sha256': reference_sha,
            '_reference_record': reference,
        })
    return prepared, rejected


def _prepare_output(output: Path, repo_root: Path, input_files: list[Path]) -> Path:
    if output.is_symlink():
        raise CurationError(f'Output cannot be a symlink: {output}')
    output_lexical = output.absolute()
    repo_lexical = repo_root.absolute()
    output_abs = output_lexical.resolve()
    repo_abs = repo_lexical.resolve()
    if not _under(output_abs, repo_abs):
        raise CurationError(f'Output must be inside the repository: {output}')
    _reject_symlink_components(output_lexical, repo_lexical)
    if output_abs.exists():
        if not output_abs.is_dir() or any(output_abs.iterdir()):
            raise CurationError(f'Refusing to overwrite nonempty output: {output_abs}')
    for source in input_files:
        if _under(source.resolve(), output_abs.resolve()) or _under(output_abs.resolve(), source.resolve()):
            raise CurationError(f'Output and input source overlap: {source}')
    output_abs.mkdir(parents=True, exist_ok=True)
    return output_abs


def _copy_recognition(
    output: Path,
    references: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
    review: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    gallery_records = []
    for row in references:
        game, language, card_id = row['_identity']
        source = row['_resolved_path']
        digest = row['_sha256']
        destination = output / 'recognition' / 'gallery' / game / language / card_id / f'reference-{digest[:12]}{source.suffix.lower()}'
        _copy_file(source, destination, mode)
        _write_json(destination.with_suffix(destination.suffix + '.json'), {
            'artifact_type': 'recognition_gallery_reference', 'source_path': row['path'], 'source_sha256': digest,
            'identity': {'game': game, 'language': language, 'card_id': card_id},
            'source_record': {k: v for k, v in row.items() if not k.startswith('_')},
        })
        gallery_records.append({'image': destination.relative_to(output).as_posix(), 'sha256': digest,
                                'identity': {'game': game, 'language': language, 'card_id': card_id}})

    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in sorted(SPLITS)}
    for row in prepared:
        game, language, card_id = row['_identity']
        split, source, digest = row['_split'], row['_crop'], row['_crop_sha256']
        destination = output / 'recognition' / split / game / language / card_id / f'query-{digest[:16]}{source.suffix.lower()}'
        _copy_file(source, destination, mode)
        identity = {'game': game, 'language': language, 'card_id': card_id}
        sidecar = {
            'artifact_type': 'recognition_query', 'image': destination.relative_to(output).as_posix(),
            'crop_sha256': digest, 'source_path': row['source_path'], 'source_sha256': row['_source_sha256'],
            'source_group': row['source_group'], 'split': split, 'status': row['status'],
            'single_complete_card': True, 'identity': identity,
            'reference_path': row['reference_path'], 'reference_sha256': row['_reference_sha256'],
            'annotation_provenance': row['annotation_provenance'], 'annotation_notes': row['annotation_notes'],
            'reviewer': review['reviewer'], 'review_provenance': review['review_provenance'],
            'source_record': {k: v for k, v in row.items() if not k.startswith('_')},
        }
        _write_json(destination.with_suffix(destination.suffix + '.json'), sidecar)
        split_records[split].append({'image': destination.relative_to(output).as_posix(), 'sha256': digest,
                                     'source_group': row['source_group'], 'identity': identity})

    manifest = {
        'schema_version': 1, 'artifact_type': 'curated_recognition_dataset',
        'copy_mode': mode, 'identity_key_fields': ['game', 'language', 'card_id'],
        'gallery_reference_count': len(gallery_records), 'gallery': gallery_records,
        'splits': {split: {'query_count': len(rows), 'queries': rows} for split, rows in split_records.items()},
        'source_group_split_isolation_enforced': True,
        'evaluation_gallery_byte_overlap_rejected': True,
        'label_policy': 'root-reviewed verified_reference_match only; the curation tool does not assign identity',
        'reviewer': review['reviewer'], 'review_provenance': review['review_provenance'],
    }
    _write_json(output / 'recognition' / 'manifest.json', manifest)
    return manifest


def _validate_obb_text(label_path: Path, expected_instances: int) -> None:
    try:
        lines = [line.split() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except OSError as exc:
        raise CurationError(f'Cannot read synthetic OBB labels: {label_path}') from exc
    if len(lines) != expected_instances:
        raise CurationError(f'OBB instance count disagrees with annotation: {label_path}')
    for line in lines:
        if len(line) != 9 or line[0] != '0':
            raise CurationError(f'Expected class-0 YOLO OBB row with eight coordinates: {label_path}')
        try:
            values = [float(x) for x in line[1:]]
        except ValueError as exc:
            raise CurationError(f'Non-numeric YOLO OBB row: {label_path}') from exc
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise CurationError(f'Invalid normalized YOLO OBB coordinates: {label_path}')


def _validate_synthetic_annotation(annotation: dict[str, Any], split: str, image_path: Path, label_path: Path) -> None:
    if annotation.get('label_origin') != 'synthetic_renderer' or annotation.get('real_camera_validation') is not False:
        raise CurationError('Only renderer-labeled synthetic scenes may enter the synthetic detector split')
    if annotation.get('split') != split:
        raise CurationError(f'Synthetic annotation split mismatch: {image_path}')
    if not isinstance(annotation.get('instances'), list) or annotation.get('count') != len(annotation['instances']):
        raise CurationError(f'Synthetic annotation count mismatch: {image_path}')
    width, height = annotation.get('width'), annotation.get('height')
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
        raise CurationError(f'Synthetic annotation has invalid dimensions: {image_path}')
    actual_width, actual_height, _ = verify_image(image_path)
    if (actual_width, actual_height) != (width, height):
        raise CurationError(f'Synthetic image dimensions do not match annotation: {image_path}')
    for instance in annotation['instances']:
        for field in ('true_corners', 'obb_corners'):
            points = instance.get(field)
            if not isinstance(points, list) or len(points) != 4 or any(not isinstance(p, list) or len(p) != 2 for p in points):
                raise CurationError(f'Synthetic instance needs four {field}: {image_path}')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for point in points for v in point):
                raise CurationError(f'Synthetic instance has invalid {field}: {image_path}')
    _validate_obb_text(label_path, len(annotation['instances']))


def _copy_synthetic_detector(source_root: Path, output: Path, mode: str, repo_root: Path = ROOT) -> dict[str, Any]:
    source_root = source_root.absolute()
    if source_root.is_symlink():
        raise CurationError('Synthetic detector root cannot be a symlink')
    expected_parent = (repo_root / 'runs' / 'card_detector').absolute()
    _reject_symlink_components(source_root, expected_parent)
    source_root = source_root.resolve(strict=True)
    if not _under(source_root, expected_parent.resolve()) or source_root == expected_parent.resolve():
        raise CurationError('Synthetic detector root must be a run below runs/card_detector')
    if source_root.name != 'synthetic':
        raise CurationError('Synthetic detector import currently accepts only a directory named synthetic')
    records = []
    for split in ('train', 'val'):
        annotation_dir = source_root / 'annotations' / split
        if not annotation_dir.is_dir():
            raise CurationError(f'Missing synthetic annotations: {annotation_dir}')
        for annotation_file in sorted(annotation_dir.glob('*.json')):
            _reject_symlink_components(annotation_file, source_root)
            annotation = _json_file(annotation_file)
            source_image_rel = annotation.get('image')
            expected_prefix = Path('images') / split
            image_rel = Path(source_image_rel) if isinstance(source_image_rel, str) else Path('..')
            if image_rel.is_absolute() or image_rel.parts[:2] != expected_prefix.parts or '..' in image_rel.parts:
                raise CurationError(f'Unexpected synthetic image path in annotation: {source_image_rel!r}')
            label_rel = Path('labels') / split / f'{annotation_file.stem}.txt'
            source_image = resolve_input_file(source_image_rel, source_root, image=True)
            source_label = resolve_input_file(label_rel.as_posix(), source_root)
            _validate_synthetic_annotation(annotation, split, source_image, source_label)
            destination_image = output / 'detection' / split / 'images' / source_image.name
            destination_label = output / 'detection' / split / 'labels' / source_label.name
            destination_annotation = output / 'detection' / split / 'annotations' / annotation_file.name
            for path in (destination_image, destination_label, destination_annotation):
                if path.exists():
                    raise CurationError(f'Synthetic scene name collision: {path}')
            _copy_file(source_image, destination_image, mode)
            _copy_file(source_label, destination_label, mode)
            # Rebase paths without editing the renderer's original or a hardlink.
            _write_json(destination_annotation, {
                **annotation, 'image': f'{split}/images/{source_image.name}',
                'label': f'{split}/labels/{source_label.name}',
                'source_annotation': str(annotation_file.relative_to(repo_root.resolve())),
                'source_annotation_sha256': sha256_file(annotation_file),
            })
            records.append({
                'split': split, 'image': destination_image.relative_to(output).as_posix(),
                'image_sha256': sha256_file(destination_image), 'label': destination_label.relative_to(output).as_posix(),
                'label_sha256': sha256_file(destination_label),
                'annotation': destination_annotation.relative_to(output).as_posix(),
                'annotation_sha256': sha256_file(destination_annotation),
                'instance_count': len(annotation['instances']),
            })
    for split in sorted(SPLITS):
        for kind in ('images', 'labels', 'annotations'):
            (output / 'detection' / split / kind).mkdir(parents=True, exist_ok=True)
    (output / 'detection' / 'dataset.yaml').write_text(
        f'path: {json.dumps(str((output / "detection").resolve()))}\n'
        'train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: card\n', encoding='utf-8')
    manifest = {
        'schema_version': 1, 'artifact_type': 'curated_synthetic_detection_dataset',
        'label_origin': 'synthetic_renderer', 'source_root': str(source_root),
        'real_camera_validation': False,
        'geometry': 'true renderer quadrilaterals with separately stored derived minimum-area OBB labels',
        'dataset_yaml': 'dataset.yaml', 'scene_count': len(records),
        'split_counts': {split: sum(row['split'] == split for row in records) for split in ('train', 'val')},
        'scenes': records,
    }
    _write_json(output / 'detection' / 'manifest.json', manifest)
    return manifest


def _validate_rejected_crop_root(repo_root: Path, crop_root: Path) -> Path:
    if crop_root.is_symlink():
        raise CurationError('Rejected-crop root cannot be a symlink')
    root = crop_root.resolve(strict=True)
    repo = repo_root.resolve()
    if not _under(root, repo):
        raise CurationError('Rejected-crop root must be inside the repository')
    rel = root.relative_to(repo)
    allowed = (
        len(rel.parts) >= 4 and rel.parts[0] == 'runs' and rel.parts[1] == 'video_mining' and rel.parts[-1] in {'crops', 'detected_cards'},
        len(rel.parts) >= 4 and rel.parts[0] == 'runs' and rel.parts[1] == 'card_detector' and rel.parts[-1] == 'recognizer_crops',
    )
    if not any(allowed):
        raise CurationError(f'Rejected-crop deletion is limited to allowlisted ignored crop roots under runs/: {root}')
    _reject_symlink_components(root, repo)
    return root


def _remove_rejected(
    rejected: list[dict[str, Any]], repo_root: Path, crop_root: Path, accepted_output: Path,
) -> int:
    root = _validate_rejected_crop_root(repo_root, crop_root)
    if not accepted_output.exists() or not any(p.is_file() for p in accepted_output.rglob('*')):
        raise CurationError('Accepted copies must exist before rejected derived crops may be removed')
    targets: list[Path] = []
    for row in rejected:
        if row.get('status') != 'rejected':
            continue
        relative = Path(row.get('crop_path', ''))
        if relative.is_absolute() or not relative.parts or any(p in {'.', '..'} for p in relative.parts):
            raise CurationError('Rejected crop paths must be safe relative paths under --rejected-crop-root')
        target = root / relative
        _reject_symlink_components(target, root)
        if target.suffix.lower() not in IMAGE_SUFFIXES or not target.is_file():
            raise CurationError(f'Rejected derived crop is missing or not an image: {target}')
        _verify_sha(target, row.get('crop_sha256'), 'rejected.crop_sha256')
        targets.append(target)
    # Validate the complete deletion set before mutating anything.
    if len(set(targets)) != len(targets):
        raise CurationError('Rejected manifest contains duplicate crop paths')
    for target in targets:
        target.unlink()
    return len(targets)


def curate_dataset(
    *, repo_root: Path, input_root: Path, references_manifest: Path, review_manifest: Path,
    output: Path, copy_mode: str = 'copy', synthetic_detector: Path | None = None,
    remove_rejected_derived: bool = False, rejected_crop_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.absolute()
    input_root = input_root.absolute()
    if not repo_root.is_dir() or not input_root.is_dir():
        raise CurationError('Repository and input roots must exist as directories')
    if not _under(input_root.resolve(strict=True), repo_root.resolve(strict=True)):
        raise CurationError('Input root must be inside the repository')
    if remove_rejected_derived and rejected_crop_root is None:
        raise CurationError('--remove-rejected-derived requires --rejected-crop-root')
    if not remove_rejected_derived and rejected_crop_root is not None:
        raise CurationError('--rejected-crop-root requires --remove-rejected-derived')
    if references_manifest.is_symlink() or review_manifest.is_symlink():
        raise CurationError('Manifest files cannot be symlinks')
    _reject_symlink_components(references_manifest.absolute(), input_root)
    _reject_symlink_components(review_manifest.absolute(), input_root)
    references_manifest = references_manifest.resolve(strict=True)
    review_manifest = review_manifest.resolve(strict=True)
    if not _under(references_manifest, input_root.resolve()) or not _under(review_manifest, input_root.resolve()):
        raise CurationError('Manifest files must be inside --input-root')
    reference_rows, ref_by_key, gallery_hashes = _load_references(references_manifest, input_root)
    review = _json_file(review_manifest)
    prepared, rejected = _prepare_queries(review_manifest, review, input_root, ref_by_key, gallery_hashes)
    if remove_rejected_derived:
        assert rejected_crop_root is not None
        _validate_rejected_crop_root(repo_root, rejected_crop_root)
    relevant_sources = [references_manifest, review_manifest]
    relevant_sources.extend(row['_resolved_path'] for row in ref_by_key.values())
    relevant_sources.extend(row[key] for row in prepared for key in ('_crop', '_source', '_reference'))
    output_path = _prepare_output(output, repo_root, relevant_sources)
    recognition = _copy_recognition(output_path, reference_rows, prepared, review, copy_mode)
    detection = None
    if synthetic_detector is not None:
        source_path = synthetic_detector.resolve(strict=True)
        if _under(output_path.resolve(), source_path) or _under(source_path, output_path.resolve()):
            raise CurationError('Output and synthetic detector source cannot overlap')
        detection = _copy_synthetic_detector(source_path, output_path, copy_mode, repo_root)
    deleted = 0
    if remove_rejected_derived:
        if rejected_crop_root is None:
            raise CurationError('--remove-rejected-derived requires --rejected-crop-root')
        deleted = _remove_rejected(rejected, repo_root, rejected_crop_root, output_path)
    elif rejected_crop_root is not None:
        raise CurationError('--rejected-crop-root requires --remove-rejected-derived')
    summary = {
        'output': str(output_path), 'recognition_gallery_references': len(reference_rows),
        'recognition_queries': len(prepared), 'recognition_split_counts': {
            split: sum(row['_split'] == split for row in prepared) for split in sorted(SPLITS)},
        'synthetic_detection_scenes': detection['scene_count'] if detection else 0,
        'rejected_derived_crops_removed': deleted,
        'real_camera_validation': False,
    }
    _write_json(output_path / 'curation_manifest.json', {
        'schema_version': 1, 'artifact_type': 'curated_dataset_root', **summary,
        'recognition_manifest': 'recognition/manifest.json',
        'detection_manifest': 'detection/manifest.json' if detection else None,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--input-root', type=Path, required=True, help='Root for paths in the manifests')
    parser.add_argument('--references-manifest', type=Path, required=True)
    parser.add_argument('--review-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--copy-mode', choices=('copy', 'hardlink'), default='copy')
    parser.add_argument('--synthetic-detector', type=Path,
                        help='Optional runs/card_detector/<run>/synthetic-contract source; real scenes are never imported here')
    parser.add_argument('--remove-rejected-derived', action='store_true',
                        help='Explicitly remove only hash-verified rejected crops below an allowlisted ignored runs crop root')
    parser.add_argument('--rejected-crop-root', type=Path,
                        help='Required with --remove-rejected-derived; crop paths in rejected records are relative to it')
    args = parser.parse_args()
    try:
        summary = curate_dataset(
            repo_root=args.repo_root, input_root=args.input_root,
            references_manifest=args.references_manifest, review_manifest=args.review_manifest,
            output=args.output, copy_mode=args.copy_mode, synthetic_detector=args.synthetic_detector,
            remove_rejected_derived=args.remove_rejected_derived, rejected_crop_root=args.rejected_crop_root,
        )
    except (CurationError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
