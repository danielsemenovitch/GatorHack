"""Transforms a blot panel goes through on its way into a published paper.

Used two ways: to build the benchmark (fakes are made, then published) and
to measure robustness (the same table we filled in by hand for SynthID).
"""
from __future__ import annotations

import io
import random

import pymupdf as fitz
from PIL import Image, ImageEnhance


def resave_png(im: Image.Image) -> Image.Image:
    return Image.frombytes("RGB", im.size, im.convert("RGB").tobytes())


def jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize(im: Image.Image, f: float) -> Image.Image:
    w, h = im.size
    return im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.LANCZOS)


def contrast(im: Image.Image, c: float = 1.3, b: float = 1.05) -> Image.Image:
    return ImageEnhance.Brightness(ImageEnhance.Contrast(im).enhance(c)).enhance(b)


def pdf_roundtrip(im: Image.Image, q: int = 80) -> Image.Image:
    """Embed in a PDF as JPEG (as typesetting does) and extract it again."""
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=q)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    w, h = im.size
    scale = min(540 / w, 700 / h, 1.0)
    page.insert_image(fitz.Rect(36, 36, 36 + w * scale, 36 + h * scale), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    doc = fitz.open(stream=data, filetype="pdf")
    xref = doc[0].get_images(full=True)[0][0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    out = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return out


# Named transforms, matching the SynthID experiment where possible.
TRANSFORMS = {
    "none": lambda im: im,
    "resave": resave_png,
    "resize_50": lambda im: resize(im, 0.5),
    "jpeg_q85": lambda im: jpeg(im, 85),
    "jpeg_q60": lambda im: jpeg(im, 60),
    "contrast": contrast,
    "pdf_roundtrip": pdf_roundtrip,
}


def random_publish(im: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """Apply a random but realistic chain of publishing steps."""
    steps = []
    if rng.random() < 0.3:
        im = contrast(im, rng.uniform(1.05, 1.4), rng.uniform(0.95, 1.1)); steps.append("contrast")
    if rng.random() < 0.5:
        f = rng.uniform(0.5, 0.95)
        im = resize(im, f); steps.append(f"resize_{f:.2f}")
    r = rng.random()
    if r < 0.45:
        q = rng.randint(70, 95); im = jpeg(im, q); steps.append(f"jpeg_q{q}")
    elif r < 0.7:
        im = pdf_roundtrip(im, rng.randint(70, 90)); steps.append("pdf_roundtrip")
    else:
        steps.append("png")
    return im, steps
