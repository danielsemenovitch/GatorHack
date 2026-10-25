"""Default-model regression: V5 must actually load and score, not only be named."""
from __future__ import annotations

import inspect
import numpy as np
from PIL import Image

from btw import report
from btw.api import analyze_upload
from btw.detector import CnnDetector


def test_default_v5_is_wired_through_report_and_api():
    assert report.DEFAULT_MODEL == "v5"
    assert inspect.signature(analyze_upload).parameters["cnn_version"].default.default == "v5"


def test_v5_checkpoint_loads_and_scores_with_its_three_channel_network():
    det = CnnDetector.load("v5")
    assert det._backend.__name__ == "btw.cnn_v5"
    assert det.params["stem"]["w"].shape == (3, 3, 3, 24)
    assert det.params["res3"]["b"]["w"].shape == (3, 3, 48, 48)
    assert det.space == "logit"
    rng = np.random.default_rng(24)
    img = Image.fromarray(rng.integers(75, 210, (44, 64), dtype=np.uint8), "L").convert("RGB")
    result = det.analyze(img)
    assert 0 <= result.score <= 1 and np.isfinite(result.score)
    assert result.heat.shape == (44, 64)
    assert np.isfinite(result.heat).all()


def test_v3_remains_loadable_on_original_backend():
    det = CnnDetector.load("v3")
    assert det._backend.__name__ == "btw.cnn"
