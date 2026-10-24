"""Assess existing v2 vs trained v3 on a REAL external CC BY research-paper PDF.

This is an unlabeled sanity check (NO ground-truth image-fraud annotations),
not a measurement of model sensitivity or accuracy. A high flag count means
the detector may be over-alerting; it DOES NOT accuse the paper's authors.
Source: Nazim et al. Oncotarget 2018, DOI 10.18632/oncotarget.23647 CC BY 3.0.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import btw.report as review
from btw import cnn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_paper_cnn_comparison.json"
PDF_URL = "https://www.oncotarget.com/article/23647/pdf/"


def main():
    request = urllib.request.Request(PDF_URL, headers={
        "User-Agent": "BelowTheWatermarkResearchDemo/1.0",
        "Accept": "application/pdf",
    })
    with urllib.request.urlopen(request, timeout=35) as response:
        data = response.read(25_000_000)
    if not data.startswith(b"%PDF") or len(data) < 50_000:
        raise RuntimeError(f"Unexpected PDF response: {data[:60]!r}")
    print("Downloaded CC BY article PDF:", len(data), hashlib.sha256(data).hexdigest(), flush=True)

    v3 = Path("/tmp/v3-artifact/models/splice_cnn.npz")
    v2 = ROOT / "models" / "splice_cnn.npz"
    if not v3.exists():
        raise RuntimeError("V3 artifact was not downloaded")

    result = {}
    for label, weights in [("v2", v2), ("v3", v3)]:
        cnn.WEIGHTS = weights
        review._DETECTOR = None
        print("BEGIN", label, flush=True)
        rep = review.analyze(data, "Nazim_et_al_2018_CC_BY.pdf", max_panels=64, blot_mode="auto")
        per_figure = rep["summary"]["per_figure"]
        result[label] = {
            "summary": {k: v for k, v in rep["summary"].items()
                        if k != "widespread_flag_figures"},
            "score_kind": rep["summary"]["score_kind"],
            "flags_by_figure": [
                {"label": f["label"], "flagged": sum(bool(p["flagged"]) for p in f["panels"]),
                 "analyzed": len(f["panels"]),
                 "scores": [p["score"] for p in f["panels"]]}
                for f in rep["figures"] if f["panels"]
            ],
            "widespread_flag_figures": rep["summary"]["widespread_flag_figures"],
            "duplicate_pairs": [
                {"a": {k: v for k, v in d["a"].items() if k != "crop"},
                 "b": {k: v for k, v in d["b"].items() if k != "crop"},
                 "noise_similarity": d["noise_similarity"], "shape_similarity": d["shape_similarity"]}
                for d in rep["duplicates"]
            ],
        }
        print("MODEL", label, "screened", result[label]["summary"]["panels"], "flagged",
              result[label]["summary"]["flagged_panels"], "unassessed due cap",
              result[label]["summary"]["skipped_panels"], flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "source": PDF_URL,
        "sha256": hashlib.sha256(data).hexdigest(),
        "license": "CC BY 3.0",
        "warning": "Unlabeled publication sanity check only. Never infer manipulation or model accuracy from flags.",
        "models": result,
    }, indent=2))
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
