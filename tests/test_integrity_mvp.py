"""Regression tests for the scientific image review MVP.

Synthetic images only: no published author or figure is implicated.
"""
from __future__ import annotations

import io

import numpy as np
import pymupdf
from PIL import Image

from btw.generic_duplicates import find_image_reuse
import btw.report as report


def image(seed=12):
    y, x = np.mgrid[:180, :280]
    rng = np.random.default_rng(seed)
    g = (155 + 50 * np.sin(x / 34) + 32 * np.cos(y / 21)
         + rng.normal(0, 3, (180, 280))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([g, np.roll(g, 16, axis=1), 255 - g // 3], axis=-1), "RGB")


def encode(im, fmt="PNG", **kwargs):
    data = io.BytesIO()
    im.save(data, format=fmt, **kwargs)
    return data.getvalue()


def test_exact_pixel_copy_even_if_encodings_differ():
    a = image()
    png = Image.open(io.BytesIO(encode(a)))
    matches = find_image_reuse([a, png, image(33)])
    assert len(matches) == 1
    assert matches[0]["method"] == "exact_pixel_duplicate"
    assert matches[0]["a"]["figure"] == 0
    assert matches[0]["b"]["figure"] == 1


def test_near_duplicate_after_jpeg_resave():
    original = image()
    lossy = Image.open(io.BytesIO(encode(original, "JPEG", quality=80))).convert("RGB")
    matches = find_image_reuse([original, lossy])
    assert len(matches) == 1
    assert matches[0]["method"] == "near_duplicate"
    assert matches[0]["similarity"] >= 0.92


def test_unrelated_images_not_marked_as_reused():
    assert find_image_reuse([image(12), image(22)]) == []


def test_nonblot_upload_skips_cnn(monkeypatch):
    def forbidden():
        raise AssertionError("Blot model was incorrectly run on a nonblot image")
    monkeypatch.setattr(report, "detector", forbidden)
    result = report.analyze(encode(image()), "microscopy.png")
    assert result["summary"]["figures"] == 1
    assert result["summary"]["panels"] == 0
    assert result["checks"]["blot_cnn"] == "not_assessed"
    assert result["checks"]["whole_image_reuse"] == "not_applicable_single_image"
    assert result["summary"]["threshold"] is None
    assert result["image_reuse"] == []


def test_pdf_cross_page_reuse_and_source_labels(monkeypatch):
    def forbidden():
        raise AssertionError("Blot model was incorrectly run on a nonblot PDF")
    monkeypatch.setattr(report, "detector", forbidden)
    png = encode(image())
    pdf = pymupdf.open()
    for _ in range(2):
        page = pdf.new_page(width=420, height=300)
        page.insert_image(pymupdf.Rect(20, 20, 320, 220), stream=png)
    data = pdf.tobytes()
    pdf.close()
    result = report.analyze(data, "two-page-microscopy.pdf")
    assert len(result["figures"]) == 2
    assert result["figures"][0]["label"].startswith("page 1")
    assert result["figures"][1]["label"].startswith("page 2")
    assert result["summary"]["image_reuse_pairs"] == 1
    assert result["image_reuse"][0]["method"] == "exact_pixel_duplicate"
    assert result["checks"]["c2pa"] == "not_assessed"


def test_low_information_not_overclaimed_as_near_duplicate():
    base = Image.new("RGB", (300, 150), "white")
    modified = Image.new("RGB", (300, 150), (245, 245, 245))
    assert find_image_reuse([base, modified]) == []
