"""Splice detector: window classifier + panel and band level aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .bands import find_bands
from .features import FEATURE_NAMES, heatmap, window_features
from .imgio import to_gray

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "splice_hgb.joblib"


def panel_score(probs: np.ndarray, X: np.ndarray | None = None, method: str = "outlier", k: int = 3) -> float:
    """Image level suspicion from window probabilities.

    "top":     mean of the top k window probabilities.
    "outlier": how far the most suspicious band window sits above the
               typical band window of the SAME panel. Edits are local
               anomalies, so a panel is judged against itself.
    """
    if len(probs) == 0:
        return 0.0
    if method == "top" or X is None:
        return float(np.sort(probs)[-min(k, len(probs)):].mean())
    band = X[:, 13] > 0.15
    p = probs[band] if band.sum() >= 3 else probs
    top = float(np.sort(p)[-min(k, len(p)):].mean())
    return float(np.clip(top - np.median(p) + 0.5 * top, 0, 1.5) / 1.5)


@dataclass
class PanelResult:
    score: float
    heat: np.ndarray            # HxW float in 0..1
    bands: list[dict]           # band boxes with scores
    windows: list               # (x, y, w, h)
    probs: np.ndarray
    top_features: list[tuple[str, float]]


class Detector:
    def __init__(self, model=None, threshold: float = 0.5, score_method: str = "outlier"):
        self.model = model
        self.threshold = threshold
        self.score_method = score_method

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Detector":
        obj = joblib.load(path)
        return cls(obj["model"], obj.get("threshold", 0.5), obj.get("score_method", "outlier"))

    def save(self, path: Path = MODEL_PATH, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "score_method": self.score_method,
                     "features": FEATURE_NAMES, **extra}, path)

    def analyze(self, panel: Image.Image) -> PanelResult:
        gray = to_gray(panel)
        wins, X = window_features(gray)
        probs = self.model.predict_proba(X)[:, 1]
        heat = heatmap(wins, probs, gray.shape)
        bands = []
        for b in find_bands(gray):
            x0, y0, x1, y1 = b.box()
            pad_y = max(2, b.h // 2)
            region = heat[max(0, y0 - pad_y):y1 + pad_y, x0:x1]
            s = float(np.percentile(region, 90)) if region.size else 0.0
            bands.append({**b.to_dict(), "score": round(s, 3)})
        # Which features pushed the most suspicious window: deviation from 0
        # (features are panel relative, so 0 means "like the rest of the panel").
        i = int(np.argmax(probs)) if len(probs) else 0
        dev = sorted(zip(FEATURE_NAMES, X[i].tolist()), key=lambda t: -abs(t[1]))[:4] if len(probs) else []
        return PanelResult(panel_score(probs, X, self.score_method), heat, bands, wins, probs, dev)


class CnnDetector:
    """Same interface as Detector, backed by the JAX patch CNN."""

    def __init__(self, params, threshold: float):
        self.params = params
        self.threshold = threshold
        self.score_method = "top"

    @classmethod
    def load(cls) -> "CnnDetector":
        from . import cnn
        params, meta = cnn.load()
        return cls(params, float(meta.get("threshold", 0.5)))

    def analyze(self, panel: Image.Image) -> PanelResult:
        from . import cnn
        gray = to_gray(panel)
        X, pos, band, _ = cnn.extract(gray)
        h, w = gray.shape
        probs = cnn.predict(self.params, X) if len(X) else np.zeros(0, np.float32)
        wins = [(x, y, cnn.P, cnn.P) for x, y in pos]
        heat = heatmap(wins, probs, (max(h, cnn.P), max(w, cnn.P)))[:h, :w]
        bands = []
        for b in find_bands(gray):
            x0, y0, x1, y1 = b.box()
            pad_y = max(2, b.h // 2)
            region = heat[max(0, y0 - pad_y):y1 + pad_y, x0:x1]
            bands.append({**b.to_dict(), "score": round(float(np.percentile(region, 90)) if region.size else 0.0, 3)})
        p = probs[band] if band.sum() >= 3 else probs
        score = float(np.sort(p)[-min(3, len(p)):].mean()) if len(p) else 0.0
        # Explanations come from the hand built features at the hottest spot.
        dev = []
        try:
            fw, FX = window_features(gray)
            if len(probs):
                i = int(np.argmax(probs)); cx, cy = pos[i][0] + cnn.P // 2, pos[i][1] + cnn.P // 2
                j = int(np.argmin([abs(x + ww / 2 - cx) + abs(y + hh / 2 - cy) for x, y, ww, hh in fw]))
                dev = sorted(zip(FEATURE_NAMES, FX[j].tolist()), key=lambda t: -abs(t[1]))[:4]
        except Exception:
            pass
        return PanelResult(score, heat, bands, wins, probs, dev)


def load_best():
    """Load whichever detector measured better on the held out test papers."""
    import json
    root = Path(__file__).resolve().parents[1]
    def auc(name):
        f = root / "results" / name
        return json.loads(f.read_text()).get("image_auroc", 0) if f.exists() else 0
    if (root / "models" / "splice_cnn.npz").exists() and auc("cnn_metrics.json") >= auc("detector_metrics.json"):
        return CnnDetector.load()
    return Detector.load()
