"""Precompute reports for the demo examples shown in the app.

The hero is a published CC BY figure into which WE pasted one AI generated
band for this demo. The authors' published figure is unaltered.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from btw.report import analyze

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"

CITE_ALK2 = ("Figure from PMC10376983 (Biomolecules, CC BY 4.0). We pasted one AI generated band into the "
             "wildtype FK506 lanes for this demo. The published figure is unaltered.")

EXAMPLES = [
    # The hero image (the SynthID finding). Not offered as a clickable example:
    # our detector misses this edit and flags an untouched loading control.
    {"id": "alk2_ai_band", "file": "alk2_ai_band.jpg", "title": "Published blot with one AI band added",
     "hero": True, "credit": CITE_ALK2, "synthid_row": "S1",
     # Ground truth from examples/alk2_ai_band_mask.png, in pixels of the 739x318 image.
     "known_edit": {"x": 262, "y": 143, "w": 49, "h": 24, "image_width": 739, "image_height": 318,
                    "source": "alk2_ai_band_mask.png",
                    "note": "Region we edited for this demo (ground truth from our own mask)."}},
    {"id": "pmc10315040_ai_band", "file": "pmc10315040_fig5_ai_band.jpg",
     "title": "A pasted AI band the model catches",
     "blurb": "At Standard it catches about a third of these. This is one of them.",
     "credit": ("Fig. 5 of PMC10315040 (doi:10.1186/s13048-023-01210-5, CC BY). We pasted one AI generated band "
                "into the IL-6 strip; the published figure is unaltered. Chosen from our end to end test, where "
                "the edited panel was flagged in 7 of 22 figures."),
     # Ground truth from scripts/eval_figure_level.py (edit_box), pixels of the 708x1001 image.
     "known_edit": {"x": 548, "y": 154, "w": 19, "h": 17, "image_width": 708, "image_height": 1001,
                    "source": "pmc10315040_fig5_mask_in_panel.png",
                    "note": "Region we edited for this demo (ground truth from our own mask)."}},
    {"id": "demo_reuse", "file": "demo_reuse.pdf", "title": "A PDF with one figure used twice",
     "blurb": "Figure 3 is a re-saved copy of Figure 1.",
     "credit": ("A 4 page PDF we assembled from two CC BY figures (PMC10352194 Fig. 4, PMC10264620 Fig. 3). "
                "Figure 3 is a re-saved copy of Figure 1 on purpose; nothing here reflects anything the "
                "original authors did.")},
    {"id": "gemini_blot", "file": "gemini_whole_blot.jpeg", "title": "A whole blot made by Gemini",
     "blurb": "Straight from Gemini, credential intact.",
     "credit": "Generated with Gemini for this project. Still carries Google's content credential."},
]


def _thumb(data_url: str, width: int = 360) -> str:
    """Small JPEG of the first figure for the example cards."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    synthid = {r["id"]: r["synthid"] for r in json.loads((ROOT / "results/synthid_manual.json").read_text())["rows"]}
    out = []
    for ex in EXAMPLES:
        f = EX / ex["file"]
        if not f.exists():
            print("missing", f); continue
        rep = analyze(f.read_bytes(), ex["file"])
        rep["credit"] = ex["credit"]
        name = f"{ex['id']}.report.json"
        (EX / name).write_text(json.dumps(rep))
        entry = {k: ex[k] for k in ("id", "title", "blurb", "credit", "known_edit") if k in ex}
        entry.update({"report": name, "file": ex["file"], "hero": ex.get("hero", False),
                      "synthid": synthid.get(ex.get("synthid_row"), "n/a")})
        entry["thumb"] = _thumb(rep["figures"][0]["image"]) if rep["figures"] else None
        if ex.get("hero"):
            fig = rep["figures"][0]
            entry["preview"] = fig
            flagged = rep["summary"]["flagged_panels"]
            entry["preview_verdict"] = "flagged" if flagged else "not flagged"
        out.append(entry)
        print(ex["id"], rep["verdict"], rep["summary"])
    (EX / "examples.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
