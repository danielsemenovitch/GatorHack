"""Image loading helpers shared by every module."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(src) -> Image.Image:
    """Load an image path, bytes or PIL image as RGB, honoring EXIF rotation."""
    if isinstance(src, Image.Image):
        im = src
    elif isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(src))
    else:
        im = Image.open(Path(src))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def to_gray(im: Image.Image) -> np.ndarray:
    """Float32 grayscale in [0, 255]."""
    return np.asarray(im.convert("L"), dtype=np.float32)


def normalize_polarity(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return an image where bands are dark on a light background.

    Fluorescent blots are often light bands on a dark background. We invert
    those so every downstream step can assume dark bands.
    """
    # Bands are the minority of pixels and sit in one tail of the histogram.
    # If the bright tail is longer than the dark tail, bands are bright.
    p1, p50, p99 = np.percentile(gray, [1, 50, 99])
    inverted = (p99 - p50) > 1.3 * (p50 - p1) and p50 < 140
    return (255.0 - gray if inverted else gray), bool(inverted)


def png_bytes(arr_or_img) -> bytes:
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(
        np.clip(arr_or_img, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
