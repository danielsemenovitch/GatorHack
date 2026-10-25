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

    def __init__(self, params, threshold: float, rule: str = "top", space: str = "prob", scale: float = 1.0, backend=None):
        from . import cnn
        self._backend = backend if backend is not None else cnn
        self.params = params
        self.space = space
        self.score_method = rule
        self.scale = max(1e-3, scale)
        # In logit space every score is shown on a 0..1 display scale where
        # the flag line sits at exactly 0.5 (raw threshold kept for reference).
        self.raw_threshold = threshold
        self.threshold = 0.5 if space == "logit" else threshold

    @classmethod
    def load(cls, version: str = "v2") -> "CnnDetector":
        from . import cnn
        import re
        if not re.fullmatch(r"v\d+", version or ""):
            raise ValueError("Unsupported CNN version")
        # V5 uses a three-channel residual architecture; older checkpoints
        # retain the original two-channel model and inference path.
        if version == "v5":
            from . import cnn_v5
            backend = cnn_v5
        else:
            backend = cnn
        path = cnn.WEIGHTS if version == "v2" else cnn.WEIGHTS.with_name(f"splice_cnn_{version}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Requested CNN checkpoint unavailable: {version}")
        params, meta = backend.load(path)
        if version == "v5" and meta.get("architecture") != "v5_residual_3_channel":
            raise ValueError("V5 checkpoint architecture mismatch")
        return cls(params, float(meta.get("threshold", 0.5)), str(meta.get("rule", "top")),
                   str(meta.get("space", "prob")), float(meta.get("scale", 1.0)), backend=backend)

    def _display(self, v):
        return 1.0 / (1.0 + np.exp(-(np.asarray(v, np.float64) - self.raw_threshold) / self.scale))

    def analyze(self, panel: Image.Image) -> PanelResult:
        cnn = self._backend
        gray = to_gray(panel)
        X, pos, band, _ = cnn.extract(gray)
        h, w = gray.shape
        wins = [(x, y, cnn.P, cnn.P) for x, y in pos]
        # The panel-level top-k score uses band-rich patches only when there
        # are at least 3 of them. Heat must use that SAME eligibility mask;
        # otherwise the UI can highlight an unrelated label or background
        # patch which did not contribute to the reported panel score.
        eligible = band if band.sum() >= 3 else np.ones(len(band), dtype=bool)
        if self.space == "logit":
            z = cnn.predict(self.params, X, logits=True) if len(X) else np.zeros(0, np.float32)
            r, raw = cnn.relative(z, band, self.score_method)
            probs = self._display(r).astype(np.float32)
            heat = heatmap(wins, np.where(eligible, probs, 0.0), (max(h, cnn.P), max(w, cnn.P)))[:h, :w]
            return self._finish(gray, heat, band, wins, probs, pos, float(self._display(raw)))
        probs = cnn.predict(self.params, X) if len(X) else np.zeros(0, np.float32)
        heat = heatmap(wins, np.where(eligible, probs, 0.0), (max(h, cnn.P), max(w, cnn.P)))[:h, :w]
        pb = probs[band] if band.sum() >= 3 else probs
        if self.score_method == "outlier" and len(pb):
            heat = np.clip(heat - float(np.median(pb)), 0, 1)  # same scale as the panel score
        p = probs[band] if band.sum() >= 3 else probs
        score = float(np.sort(p)[-min(3, len(p)):].mean()) if len(p) else 0.0
        if self.score_method == "outlier" and len(p):
            score = float(np.clip(score - np.median(p), 0, 1))
        return self._finish(gray, heat, band, wins, probs, pos, score)

    def _finish(self, gray, heat, band, wins, probs, pos, score) -> PanelResult:
        cnn = self._backend
        bands = []
        for b in find_bands(gray):
            x0, y0, x1, y1 = b.box()
            pad_y = max(2, b.h // 2)
            region = heat[max(0, y0 - pad_y):y1 + pad_y, x0:x1]
            bands.append({**b.to_dict(), "score": round(float(np.percentile(region, 90)) if region.size else 0.0, 3)})
        # Explanations come from the hand built features at the hottest spot.
        dev = []
        try:
            fw, FX = window_features(gray)
            if len(probs):
                eligible = band if band.sum() >= 3 else np.ones(len(probs), dtype=bool)
                i = int(np.argmax(np.where(eligible, probs, -1.0)))
                cx, cy = pos[i][0] + cnn.P // 2, pos[i][1] + cnn.P // 2
                j = int(np.argmin([abs(x + ww / 2 - cx) + abs(y + hh / 2 - cy) for x, y, ww, hh in fw]))
                dev = sorted(zip(FEATURE_NAMES, FX[j].tolist()), key=lambda t: -abs(t[1]))[:4]
        except Exception:
            pass
        return PanelResult(score, heat, bands, wins, probs, dev)


def load_best(version: str = "v2"):
    """Use an explicitly requested, versioned CNN checkpoint.

    v3 showed better AUROC and recall on the expanded synthetic benchmark,
    but it is still experimental and its scores are not guilt probabilities.
    The legacy v2 checkpoint remains available for direct comparison.
    """
    if version not in ("v2", "v3", "v5"):
        raise ValueError("CNN version must be v2, v3 or v5")
    return CnnDetector.load(version)
