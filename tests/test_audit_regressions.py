"""Regression tests for bugs found in the Oct 24 independent audit (AUDIT_REPORT.md).

Synthetic inputs only; no published figure or author is implicated.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from PIL import Image

import btw.report as report
from btw.provenance import read_c2pa


def _png(im: Image.Image) -> bytes:
    f = io.BytesIO()
    im.save(f, "PNG")
    return f.getvalue()


def _textured(seed: int, size=(400, 200), mode="RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    g = rng.integers(40, 220, (size[1], size[0]), dtype=np.uint8)
    im = Image.fromarray(g, "L")
    return im if mode == "L" else im.convert(mode)


def _forbid_cnn(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("CNN should not run in this test")
    monkeypatch.setattr(report, "detector", forbidden)


def test_pdf_with_grayscale_image_is_analyzed_not_crashed(monkeypatch):
    """Before: Image.frombytes('RGB') on a 1-channel pixmap raised 'not enough image data'
    and the whole PDF failed with HTTP 422. Blots are often embedded as grayscale."""
    _forbid_cnn(monkeypatch)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(20, 20, 420, 220), stream=_png(_textured(1, mode="L")))
    result = report.analyze(doc.tobytes(), "gray.pdf", blot_mode="off")
    assert result["summary"]["figures"] == 1
    assert result["figures"][0]["native_width"] == 400
    assert result["skipped_images"] == []


def test_logo_on_every_page_is_one_object_not_reuse(monkeypatch):
    """Before: a header logo on N pages produced N*(N-1)/2 'exact duplicate' pairs and a
    'suspicious' verdict. Genuine reuse of one image in two figures must still be found."""
    _forbid_cnn(monkeypatch)
    logo, fig = _png(_textured(2, (300, 90))), _png(_textured(3))
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page().insert_image(pymupdf.Rect(40, 20, 240, 80), stream=logo)
    doc[1].insert_image(pymupdf.Rect(40, 150, 440, 350), stream=fig)
    doc[3].insert_image(pymupdf.Rect(80, 200, 480, 400), stream=fig)
    result = report.analyze(doc.tobytes(), "journal.pdf", blot_mode="off")
    logo_entries = [f for f in result["figures"] if f["repeated_placements"]]
    assert len(logo_entries) == 1 and len(logo_entries[0]["repeated_placements"]) == 3
    assert result["summary"]["image_reuse_pairs"] == 1  # the figure placed twice
    assert result["summary"]["repeated_object_placements"] == 3


def test_oversized_pdf_image_is_reported_not_silently_dropped(monkeypatch):
    """Before: a >12 MP embedded image was fully decoded, then silently skipped."""
    _forbid_cnn(monkeypatch)
    calls = []
    real = report._pixmap_rgb
    monkeypatch.setattr(report, "_pixmap_rgb", lambda pix: calls.append(1) or real(pix))
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 500, 500), stream=_png(Image.new("L", (4000, 3500), 255)))
    result = report.analyze(doc.tobytes(), "big.pdf")
    assert result["summary"]["figures"] == 0
    assert result["skipped_images"][0]["reason"] == "too_large"
    assert calls == []  # never decoded


def test_spoofed_c2pa_bytes_are_not_reported_as_signed():
    """Before: byte strings 'c2pa', 'jumb' and 'Google' made the report say
    'signed by Google LLC ... AI generated' without any signature check."""
    fake = b"\xff\xd8 c2pa jumb Created by Google Generative AI. Google LLC trainedAlgorithmicMedia \xff\xd9"
    r = read_c2pa(fake)
    assert r["reader"] == "byte_search_unverified"
    assert "could not be read or verified" in r["summary"]
    assert "signed by" not in r["summary"]
    assert r["ai_generated"] is None


def test_credential_declaring_ai_is_a_review_finding(monkeypatch):
    _forbid_cnn(monkeypatch)
    monkeypatch.setattr(report, "read_c2pa", lambda data, mime: {
        "present": True, "ai_generated": True, "validated": False, "integrity_ok": True,
        "summary": "Content Credentials read; the credential declares this image AI generated."})
    result = report.analyze(_png(_textured(5)), "ai.png", blot_mode="off")
    assert result["summary"]["credential_declares_ai"] is True
    assert result["verdict"] == "suspicious"


class _SizeRecordingDetector:
    threshold = 0.5
    space = "logit"

    def __init__(self):
        self.sizes = []

    def analyze(self, im):
        self.sizes.append(im.size)
        heat = np.zeros((im.height, im.width), np.float32)
        heat[:, : im.width // 2] = 0.97
        return SimpleNamespace(score=0.97, heat=heat, top_features=[],
                               bands=[{"id": 0, "x": 10, "y": 4, "w": 20, "h": 6, "score": 0.97}])


class _Panel:
    id, x, y, w, h = 0, 100, 50, 600, 120

    def to_dict(self):
        return {"id": 0, "x": self.x, "y": self.y, "w": self.w, "h": self.h, "blot_score": 0.9}


def test_high_resolution_panels_are_scored_at_training_scale(monkeypatch):
    """Before: 2,000 px PDF figures were scored at full size, far from the ~750 px web
    figures the CNN was trained on, inflating flags on unedited blots."""
    det = _SizeRecordingDetector()
    monkeypatch.setattr(report, "detector", lambda version="v3": det)
    monkeypatch.setattr(report, "find_duplicates", lambda *a, **k: [])
    monkeypatch.setattr(report, "_blot_candidates",
                        lambda fig, mode="auto": [(_Panel(), Image.new("RGB", (600, 120), "gray"))])
    big = Image.new("RGB", (2000, 800), "white")
    result = report.analyze(_png(big), "highres.png")
    fig = result["figures"][0]
    assert fig["analysis_scale"] == pytest.approx(0.375)
    assert det.sizes == [(225, 45)]  # 600x120 crop scored at 750/2000 scale
    panel = fig["panels"][0]
    # Heat and band boxes come back in ORIGINAL panel pixels for the overlay.
    heat = Image.open(io.BytesIO(__import__("base64").b64decode(panel["heat"].split(",", 1)[1])))
    assert heat.size == (600, 120)
    alpha = np.asarray(heat)[..., 3]
    assert alpha[:, 20:280].mean() > 50 and alpha[:, 320:580].mean() < 5  # hot half stays on the left
    band = panel["bands"][0]
    assert band["x"] == round(10 * 600 / 225) and band["w"] == round(20 * 600 / 225)
    assert result["summary"]["resized_for_analysis"] == 1


def test_small_figures_are_not_rescaled(monkeypatch):
    det = _SizeRecordingDetector()
    monkeypatch.setattr(report, "detector", lambda version="v3": det)
    monkeypatch.setattr(report, "find_duplicates", lambda *a, **k: [])
    monkeypatch.setattr(report, "_blot_candidates",
                        lambda fig, mode="auto": [(_Panel(), Image.new("RGB", (600, 120), "gray"))])
    result = report.analyze(_png(Image.new("RGB", (800, 400), "white")), "web.png")
    assert result["figures"][0]["analysis_scale"] == 1.0
    assert det.sizes == [(600, 120)]


def test_api_errors_are_readable_and_do_not_leak_internals():
    from fastapi.testclient import TestClient
    from btw.api import app
    client = TestClient(app)
    empty = client.post("/api/analyze", files={"file": ("x.png", b"", "image/png")})
    assert empty.status_code == 422 and "empty" in empty.json()["detail"]
    junk = client.post("/api/analyze", files={"file": ("x.png", b"hello world", "image/png")})
    assert junk.status_code == 422
    assert "BytesIO" not in junk.json()["detail"] and "readable PNG, JPEG or PDF" in junk.json()["detail"]
