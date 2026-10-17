"""Deterministic full-card geometry and encoder pixel contract."""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np


PREPROCESSING_VERSION = "tcg-full-card-v1"
PREPROCESSING_CONTRACT = {
    "version": PREPROCESSING_VERSION,
    "source": "RGB uint8; ordered physical card corners or verified full-image bounds",
    "canonical_size": [224, 312],
    "warp": "perspective; destination pixel centers span [0,width-1] x [0,height-1]",
    "warp_interpolation": "opencv INTER_LINEAR",
    "encoder_size": [224, 224],
    "encoder_resize": "aspect-preserving fit; OpenCV INTER_AREA when shrinking, INTER_LINEAR when enlarging",
    "padding": "centered RGB channel mean in [0,255] before normalization",
    "rgb": True,
    "tensor": "CHW float32",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}


def preprocessing_contract() -> dict:
    """Return a JSON-safe copy of the shared query/gallery preprocessing contract."""
    return json.loads(json.dumps(PREPROCESSING_CONTRACT))


def preprocessing_contract_hash() -> str:
    """Hash the default contract only; non-default function overrides need a separate contract."""
    raw = json.dumps(PREPROCESSING_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def canonicalize_card(
    image: np.ndarray,
    corners: np.ndarray | None = None,
    width: int = 224,
    height: int = 312,
) -> np.ndarray:
    """Rectify a full card from RGB uint8 pixels; corners are TL,TR,BR,BL in source pixels."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("image must be a non-empty HxWx3 RGB uint8 array")
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least 2")
    h, w = image.shape[:2]
    if corners is None:
        if w < 2 or h < 2:
            raise ValueError("full-image bounds require image width and height of at least 2")
        pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    else:
        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            raise ValueError("corners must be four finite source-pixel points with shape (4, 2)")
        # Ordered convexity checks reject bow-ties, collinear points, and near-zero edges/area.
        edges = np.roll(pts, -1, axis=0) - pts
        crosses = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.any(np.linalg.norm(edges, axis=1) < 1.0) or not (np.all(crosses > 1e-3) or np.all(crosses < -1e-3)):
            raise ValueError("corners must form a non-degenerate convex quadrilateral in TL,TR,BR,BL order")
        if abs(cv2.contourArea(pts)) < 4.0:
            raise ValueError("corner quadrilateral area is too small")
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR)


def prepare_encoder_input(
    image: np.ndarray,
    size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Aspect-fit RGB card pixels, mean-pad, normalize exactly, and return CHW float32."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("image must be a non-empty HxWx3 RGB uint8 array")
    if size < 1:
        raise ValueError("size must be positive")
    mean_a, std_a = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if mean_a.shape != (3,) or std_a.shape != (3,) or not np.isfinite(mean_a).all() or not np.isfinite(std_a).all() or np.any(std_a <= 0):
        raise ValueError("mean and std must contain three finite channels; std must be positive")
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (rw, rh), interpolation=interp).astype(np.float32)
    pixels = np.broadcast_to(mean_a * 255.0, (size, size, 3)).copy()
    x, y = (size - rw) // 2, (size - rh) // 2
    pixels[y:y + rh, x:x + rw] = resized
    tensor = (pixels / 255.0 - mean_a) / std_a
    return np.transpose(tensor, (2, 0, 1)).astype(np.float32, copy=False)
