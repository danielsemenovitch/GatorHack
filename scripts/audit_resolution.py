"""Audit (Oct 24): does figure resolution drive CNN v3 false flags?

CNN v3 was trained on Europe PMC web figures (~750 px wide). Publisher PDFs
embed figures at 1,000 to 2,000+ px. This script measures, on held-out
benchmark TEST papers (never used to train v3):

  native       the test image as published on the web (training scale)
  up2.5        the same image enlarged 2.5x (what a high-res PDF figure looks like)
  up2.5->norm  enlarged, then resized back to training scale (the app's fix)

at the research threshold (0.50) and the conservative threshold (0.90).
Optionally (--pdf) it also scores every routed blot panel of one PDF at full
resolution and at training scale. Writes results/audit_resolution.json.

Run:  python -m scripts.audit_resolution [--pdf paper.pdf] [--every 2]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from btw import cnn
from btw.detector import CnnDetector
from btw.imgio import load_rgb, to_gray

ROOT = Path(__file__).resolve().parents[1]


def scorer(det):
    def score(im):
        X, _, band, _ = cnn.extract(to_gray(im))
        if not len(X):
            return 0.0
        _, raw = cnn.relative(cnn.predict(det.params, X, logits=True), band, det.score_method)
        return float(det._display(raw))
    return score


def benchmark(det, every):
    score = scorer(det)
    ix = json.loads((ROOT / "data/benchmark/index.json").read_text())
    rows = [r for r in ix if r["split"] == "test" and r["method"] in ("clean", "ai_splice", "intensify", "cross_splice")][::every]
    out = {"native": [], "up2.5": [], "up2.5->norm": []}
    labels = []
    for r in rows:
        im = load_rgb(ROOT / "data/benchmark/images" / f"{r['id']}.png")
        up = im.resize((round(im.width * 2.5), round(im.height * 2.5)), Image.BICUBIC)
        out["native"].append(score(im))
        out["up2.5"].append(score(up))
        out["up2.5->norm"].append(score(up.resize(im.size, Image.LANCZOS)))
        labels.append(r["method"])
    labels = np.array(labels)
    res = {"n": {m: int((labels == m).sum()) for m in sorted(set(labels))},
           "n_test_papers": len({r["pmcid"] for r in rows})}
    for k, v in out.items():
        v = np.array(v)
        res[k] = {"auroc": round(float(roc_auc_score((labels != "clean").astype(int), v)), 3)}
        for t in (0.5, 0.9):
            res[k][f"clean_flagged@{t}"] = round(float((v[labels == "clean"] > t).mean()), 3)
            for m in ("ai_splice", "intensify", "cross_splice"):
                res[k][f"{m}_caught@{t}"] = round(float((v[labels == m] > t).mean()), 3)
    return res


def pdf_panels(det, path):
    from btw.report import ANALYSIS_WIDTH, _blot_candidates, _figures_from_upload
    score = scorer(det)
    figs, _, _ = _figures_from_upload(Path(path).read_bytes(), "x.pdf")
    nat, norm = [], []
    for _, fig in figs:
        s = ANALYSIS_WIDTH / fig.width if fig.width > ANALYSIS_WIDTH * 1.25 else 1.0
        for _, crop in _blot_candidates(fig, "auto"):
            nat.append(score(crop))
            norm.append(score(crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)))
    nat, norm = np.array(nat), np.array(norm)
    return {"panels": len(nat), "figure_widths": [f.width for _, f in figs],
            **{f"flagged@{t}_full_resolution": int((nat > t).sum()) for t in (0.5, 0.9)},
            **{f"flagged@{t}_training_scale": int((norm > t).sum()) for t in (0.5, 0.9)},
            "median_score_full_resolution": round(float(np.median(nat)), 3) if len(nat) else None,
            "median_score_training_scale": round(float(np.median(norm)), 3) if len(norm) else None,
            "note": "Unlabeled real paper: flags are not known to be false, but the score shift tracks resolution alone."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    det = CnnDetector.load("v3")
    widths = [Image.open(f).width for f in glob.glob(str(ROOT / "data/real_figures/*.jpg"))]
    out = {"model": "v3", "training_figure_width_px": {"median": int(np.median(widths)),
                                                       "p10": int(np.percentile(widths, 10)), "p90": int(np.percentile(widths, 90))},
           "benchmark": benchmark(det, a.every)}
    if a.pdf:
        out["pdf"] = {"file": a.pdf.name, **pdf_panels(det, a.pdf)}
    (ROOT / "results/audit_resolution.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
