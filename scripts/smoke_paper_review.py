"""End-to-end scientific image integrity smoke demo, using only CC BY repo data.

A small paper PDF is CREATED BY US by repeating the SAME published figure
on two pages; that is deliberate synthetic reuse for evaluation and does not
suggest the original authors reused images improperly.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from btw.report import analyze

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "integrity_smoke"
SOURCE_A = ROOT / "data" / "real_figures" / "PMC10015710_Fig2.jpg"
SOURCE_B = ROOT / "data" / "real_figures" / "PMC10033651_Fig5.jpg"


def create_controlled_pdf():
    doc = fitz.open()
    originals = [SOURCE_A.read_bytes(), SOURCE_A.read_bytes(), SOURCE_B.read_bytes()]
    for i, data in enumerate(originals):
        page = doc.new_page(width=620, height=790)
        pix = Image.open(io.BytesIO(data))
        scale = min(560 / pix.width, 650 / pix.height)
        w, h = pix.width * scale, pix.height * scale
        page.insert_image(fitz.Rect(30, 50, 30 + w, 50 + h), stream=data)
        page.insert_text((30, 32), f"Synthetic image reuse control {i+1}", fontsize=11)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = create_controlled_pdf()
    (OUT / "controlled_reuse_demo.pdf").write_bytes(pdf)
    report = analyze(pdf, "controlled_reuse_demo.pdf", blot_mode="off")
    assert len(report["figures"]) == 3, report["summary"]
    assert report["checks"]["blot_cnn"] == "not_assessed"
    assert report["checks"]["c2pa"] == "not_assessed"
    assert report["checks"]["whole_image_reuse"] == "completed"
    exact = [r for r in report["image_reuse"] if r["method"] == "exact_pixel_duplicate"]
    assert len(exact) >= 1, report["image_reuse"]
    assert any({x["a"]["figure"], x["b"]["figure"]} == {0, 1} for x in exact)
    assert report["summary"]["image_reuse_pairs"] >= 1
    assert report["summary"]["panels"] == 0
    assert report["verdict"] == "suspicious"
    (OUT / "controlled_reuse_report.json").write_text(json.dumps(report, indent=2))
    status = {"case": "2 intentionally repeated CC BY figures + one distinct figure",
              "attribution": [SOURCE_A.name, SOURCE_B.name],
              "summary": report["summary"], "checks": report["checks"],
              "matches": report["image_reuse"],
              "report_size_bytes": (OUT / "controlled_reuse_report.json").stat().st_size}
    (OUT / "smoke_result.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    print("SUCCESS: end-to-end PDF -> extraction -> whole-image reuse -> report JSON", flush=True)


if __name__ == "__main__":
    main()
