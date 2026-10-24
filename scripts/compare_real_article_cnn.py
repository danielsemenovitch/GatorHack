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
import urllib.parse
from xml.etree import ElementTree as ET
from pathlib import Path

import io
import pymupdf as fitz
from PIL import Image

import btw.report as review
from btw import cnn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_paper_cnn_comparison.json"
PDF_URL = "https://www.oncotarget.com/article/23647/pdf/"
PMC_ID = "PMC5790529"
CANDIDATE_PDFS = [
    "https://europepmc.org/articles/PMC5790529?pdf=render",
    "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC5790529&blobtype=pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5790529/pdf/oncotarget-09-4161.pdf",
    PDF_URL,
]

def get(url, timeout=20):
    request = urllib.request.Request(url, headers={
        "User-Agent": "BelowTheWatermarkResearch/1.0",
        "Accept": "application/pdf,image/jpeg,application/xml,*/*",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(25_000_000)

def article_pdf():
    errors = []
    for url in CANDIDATE_PDFS:
        try:
            data = get(url, timeout=12)
            if data.startswith(b"%PDF") and len(data) > 50_000:
                return data, url, "original_public_pdf"
            errors.append(f"{url}: unexpected {data[:20]!r}")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:100]}")
    print("PDF source attempts failed:", errors, flush=True)
    # Fallback uses the journal's original CC BY figure JPEGs. It is a useful
    # real-figure sanity check but is NOT pixel-equivalent to the uploaded PDF.
    xml = get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5790529/fullTextXML")
    root = ET.fromstring(xml)
    doc = fitz.open()
    assets = 0
    for fig in root.findall(".//fig"):
        graphic = fig.find(".//graphic")
        if graphic is None:
            continue
        href = graphic.get("{http://www.w3.org/1999/xlink}href") or graphic.get("href")
        if not href:
            continue
        fname = href if href.lower().endswith(".jpg") else href + ".jpg"
        url = "https://europepmc.org/api/fulltextRepo?" + urllib.parse.urlencode({
            "pmcId": PMC_ID, "type": "FILE", "fileName": fname, "mimeType": "image/jpeg"})
        try:
            raw = get(url)
            with Image.open(io.BytesIO(raw)) as im:
                width, height = im.size
            page = doc.new_page(width=max(600, width+40), height=max(600, height+40))
            page.insert_image(fitz.Rect(20,20,20+width,20+height),stream=raw)
            assets += 1
        except Exception as exc:
            print("Failed fallback figure", href, str(exc)[:100], flush=True)
    if assets < 7:
        raise RuntimeError(f"Cannot reconstruct enough figures: {assets}")
    data = doc.tobytes()
    doc.close()
    return data, "Europe PMC CC BY original-figure JPEGs", "reconstructed_figures_not_identical_to_pdf"


def main():
    data, source, source_kind = article_pdf()
    if not data.startswith(b"%PDF") or len(data) < 50_000:
        raise RuntimeError("Failed to prepare CC BY journal figures")
    print("Analysis input:", source_kind, source, len(data),
          hashlib.sha256(data).hexdigest(), flush=True)

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
        "source": source,
        "source_kind": source_kind,
        "sha256": hashlib.sha256(data).hexdigest(),
        "license": "CC BY 3.0",
        "warning": "Unlabeled publication sanity check only. Never infer manipulation or model accuracy from flags. Reconstructed figure PDF is not byte-identical to user's original.",
        "models": result,
    }, indent=2))
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
