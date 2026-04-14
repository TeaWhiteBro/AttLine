from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def pseudo_color(arr: np.ndarray) -> np.ndarray:
    x = normalize_heatmap(arr)
    r = np.clip(1.5 * x, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - x), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def save_heatmap_image(
    arr: np.ndarray,
    path: str | Path,
    strip_height: int = 32,
    upscale: int = 1,
) -> str:
    path = str(path)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = np.repeat(arr[None, :], strip_height, axis=0)
    colored = pseudo_color(arr)
    img = Image.fromarray(colored)
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.BILINEAR)
    img.save(path)
    return path


def save_overlay_image(
    base_image: Image.Image,
    heatmap: np.ndarray,
    path: str | Path,
    alpha: float = 0.45,
    upscale: int = 1,
) -> str:
    path = str(path)
    if heatmap.ndim == 1:
        raise ValueError("Overlay only supports spatial heatmaps.")

    heat_h, heat_w = heatmap.shape
    out_w, out_h = heat_w * upscale, heat_h * upscale
    base = base_image.convert("RGB").resize((out_w, out_h), Image.BILINEAR)
    base_np = np.asarray(base).astype(np.float32) / 255.0
    color = Image.fromarray(pseudo_color(heatmap))
    if upscale > 1:
        color = color.resize((out_w, out_h), Image.BILINEAR)
    color_np = np.asarray(color).astype(np.float32) / 255.0
    mixed = np.clip((1.0 - alpha) * base_np + alpha * color_np, 0.0, 1.0)
    out = Image.fromarray((mixed * 255.0).astype(np.uint8))
    out.save(path)
    return path