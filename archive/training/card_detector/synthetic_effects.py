"""Procedural scene backgrounds and camera-like image effects for bootstrap data.

These effects are deliberately approximate. They create visual variation for
training experiments; they do not model a calibrated camera or establish field accuracy.
"""
from __future__ import annotations

import io
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int = 48) -> np.ndarray:
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def make_background(
    width: int,
    height: int,
    rng: np.random.Generator,
    style: str,
    pocket_rects: Iterable[tuple[int, int, int, int]] = (),
) -> np.ndarray:
    """Return an RGB uint8 procedural surface; rectangles are source-image pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if style not in {"binder", "desk", "plain"}:
        raise ValueError("style must be 'binder', 'desk', or 'plain'")
    h, w = height, width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if style == "binder":
        base = np.array([218, 222, 226], dtype=np.float32)
        tint = rng.uniform(-9, 9, 3)
        arr = np.broadcast_to(base + tint, (h, w, 3)).copy()
        # Vinyl grain and gentle page shading.
        grain = _smooth_noise(rng, h, w, 32) * 2.5 + rng.normal(0, 1.2, (h, w))
        page = np.clip(1.0 - 0.10 * np.abs(xx / max(w - 1, 1) - 0.5), 0.78, 1.0)
        arr *= page[..., None]
        arr += grain[..., None] * np.array([1.0, 1.0, 0.95])
        out = np.clip(arr, 0, 255).astype(np.uint8)
        # Pocket shapes imply empty clear sleeves; no card art or text is added.
        for rect in pocket_rects:
            x0, y0, x1, y1 = (int(v) for v in rect)
            x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
            y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            cv2.rectangle(out, (x0, y0), (x1, y1), (195, 202, 208), 2, cv2.LINE_AA)
            cv2.rectangle(out, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (229, 234, 237), 1, cv2.LINE_AA)
            cv2.line(out, (x0 + 5, y0 + 5), (x1 - 7, y0 + 5), (244, 246, 247), 2, cv2.LINE_AA)
            # Soft transparent-plastic sheen along one edge.
            cv2.line(out, (x0 + 5, y0 + 9), (x0 + 5, y1 - 8), (237, 241, 243), 1, cv2.LINE_AA)
        # Binder spine / page seam.
        seam_x = int(w * rng.uniform(.035, .075))
        cv2.line(out, (seam_x, 0), (seam_x, h), (176, 184, 191), max(1, w // 500), cv2.LINE_AA)
        return out

    if style == "desk":
        tone = rng.uniform([105, 67, 39], [177, 127, 83]).astype(np.float32)
        # Directional broad wood-like bands with low-frequency grain.
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.012, 0.045)
        bands = np.sin(xx * freq + np.sin(yy * freq * .23 + phase) * 1.8) * rng.uniform(3, 10)
        grain = _smooth_noise(rng, h, w, 22) * 7 + rng.normal(0, 2, (h, w))
        arr = tone + (bands + grain)[..., None] * np.array([1.0, .78, .55])
    else:
        tone = rng.uniform([190, 190, 185], [242, 242, 238]).astype(np.float32)
        shade = _smooth_noise(rng, h, w, 90) * 2.8 + rng.normal(0, .8, (h, w))
        arr = tone + shade[..., None]
    # Mild illumination falloff shared by desk and plain surfaces.
    falloff = 1.0 - .08 * (((xx / max(w, 1) - .5) ** 2 + (yy / max(h, 1) - .5) ** 2) * 2)
    return np.clip(arr * falloff[..., None], 0, 255).astype(np.uint8)


def apply_camera_effects(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Apply mild illumination, reflection, sensor, focus, and JPEG artifacts."""
    src = np.asarray(image)
    if src.ndim != 3 or src.shape[2] != 3 or src.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 RGB uint8 array")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be a finite nonnegative number")
    # Work from a copy; effect order never changes spatial geometry.
    out = src.copy().astype(np.float32)
    h, w = out.shape[:2]
    illumination = float(rng.uniform(.82, 1.12) * strength + (1 - strength))
    out *= illumination
    gradient_x = rng.uniform(-.10, .10) * strength
    gradient_y = rng.uniform(-.10, .10) * strength
    gx = np.linspace(-.5, .5, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(-.5, .5, h, dtype=np.float32)[:, None, None]
    out *= np.clip(1 + gradient_x * gx + gradient_y * gy, .65, 1.35)

    glare_alpha = 0.0
    glare_center: tuple[int, int] | None = None
    if rng.random() < min(.85, .42 * strength):
        glare_alpha = float(rng.uniform(.08, .27) * strength)
        cx, cy = int(rng.uniform(-.1, 1.1) * w), int(rng.uniform(-.1, 1.1) * h)
        glare_center = (cx, cy)
        rx, ry = max(2, int(w * rng.uniform(.12, .34))), max(2, int(h * rng.uniform(.025, .10)))
        yy, xx = np.ogrid[:h, :w]
        theta = rng.uniform(-.8, .8)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        mask = np.exp(-2.6 * ((u / rx) ** 2 + (v / ry) ** 2)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, min(w, h) * .008))
        out += (255 - out) * (mask[..., None] * glare_alpha)

    noise_sigma = float(rng.uniform(0.5, 3.5) * strength)
    if noise_sigma:
        out += rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    blur_sigma = float(rng.uniform(0, .85) * strength)
    blur_kernel = 0
    if blur_sigma >= .18:
        blur_kernel = max(3, int(round(blur_sigma * 4)) | 1)
        out = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), blur_sigma)

    jpeg_quality = int(np.clip(round(98 - rng.uniform(0, 13) * strength), 65, 100))
    if strength > 0:
        buf = io.BytesIO()
        Image.fromarray(out, mode="RGB").save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8).copy()
    params = {
        "strength": float(strength),
        "illumination_scale": illumination,
        "illumination_gradient_x": float(gradient_x),
        "illumination_gradient_y": float(gradient_y),
        "glare_alpha": glare_alpha,
        "glare_center_xy": list(glare_center) if glare_center else None,
        "noise_sigma": noise_sigma,
        "blur_sigma": blur_sigma if blur_kernel else 0.0,
        "jpeg_quality": jpeg_quality if strength > 0 else None,
    }
    return out, params
