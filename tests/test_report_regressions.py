"""Regression tests for old Oncotarget report failure modes, using safe synthetic panels."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

import btw.report as review


def png():
    im = Image.new("RGB", (180, 120), "#bbbbbb")
    f = io.BytesIO()
    im.save(f, "PNG")
    return f.getvalue()


class PanelStub:
    def __init__(self, i):
        self.id = i
        self.x, self.y = 10 + 3*i, 10
        self.w, self.h = 75, 25

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y, "w": self.w, "h": self.h, "blot_score": 0.99}


class MockCNN:
    threshold = 0.9965536892414093
    space = "prob"

    def analyze(self, im):
        return SimpleNamespace(score=0.9970422, bands=[{"id": 0, "x": 2, "y": 2, "w": 12, "h": 7, "score": 0.9994}],
                               heat=np.ones((im.height, im.width), np.float32),
                               top_features=[])


def fake_candidates(number):
    return lambda fig, mode="auto": [(PanelStub(i), Image.new("RGB", (75, 25), "gray"))
                                    for i in range(number)]


def test_retains_score_precision_and_records_skipped_candidates(monkeypatch):
    monkeypatch.setattr(review, "_blot_candidates", fake_candidates(5))
    monkeypatch.setattr(review, "detector", lambda version="v3": MockCNN())
    monkeypatch.setattr(review, "find_duplicates", lambda _, **kwargs: [])
    result = review.analyze(png(), "synth.png", max_panels=2)
    assert result["summary"]["panels"] == 2
    assert result["summary"]["skipped_panels"] == 3
    assert result["summary"]["threshold"] == 0.996554
    assert result["summary"]["max_splice_score"] == 0.997042
    assert result["figures"][0]["blot_screen"] == "partial"
    assert result["checks"]["blot_cnn"] == "partially_assessed"
    assert result["summary"]["score_is_probability"] is False


def test_many_flags_are_marked_as_correlated_candidates(monkeypatch):
    monkeypatch.setattr(review, "_blot_candidates", fake_candidates(5))
    monkeypatch.setattr(review, "detector", lambda version="v3": MockCNN())
    monkeypatch.setattr(review, "find_duplicates", lambda _, **kwargs: [])
    result = review.analyze(png(), "synth.png")
    assert result["summary"]["skipped_panels"] == 0
    assert len(result["summary"]["widespread_flag_figures"]) == 1
    assert result["summary"]["widespread_flag_figures"][0]["flagged_panels"] == 5


def test_duplicate_contains_two_evidence_crops_not_band_zero(monkeypatch):
    monkeypatch.setattr(review, "_blot_candidates", fake_candidates(2))
    monkeypatch.setattr(review, "detector", lambda version="v3": MockCNN())
    monkeypatch.setattr(review, "find_duplicates", lambda _, **kwargs: [{
        "a": {"panel": 0, "id": 4, "x": 2, "y": 2, "w": 13, "h": 10},
        "b": {"panel": 1, "id": -1, "x": 8, "y": 3, "w": 13, "h": 10},
        "transform": "none", "noise_similarity": 0.81, "shape_similarity": 0.86,
    }])
    r = review.analyze(png(), "synth.png")
    d = r["duplicates"][0]
    assert d["a"]["crop"].startswith("data:image/png;base64,")
    assert d["b"]["crop"].startswith("data:image/png;base64,")
    assert d["a"]["is_detected_band"] is True
    assert d["b"]["is_detected_band"] is False
    assert "figure_label" in d["b"]


def test_panel_limit_validation():
    import pytest
    with pytest.raises(ValueError, match="max_panels"):
        review.analyze(png(), "synthetic.png", max_panels=0)
