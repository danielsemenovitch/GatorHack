"""Precompute reports for the demo examples shown in the app.

The hero is a published CC BY figure into which WE pasted one AI generated
band for this demo. The authors' published figure is unaltered.
"""
from __future__ import annotations

import json
from pathlib import Path

from btw.report import analyze

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"

CITE_ALK2 = ("Figure from PMC10376983 (Biomolecules, CC BY 4.0). We pasted one AI generated band into the "
             "wildtype FK506 lanes for this demo. The published figure is unaltered.")

EXAMPLES = [
    {"id": "alk2_ai_band", "file": "alk2_ai_band.jpg", "title": "Published blot with one AI band added",
     "hero": True, "credit": CITE_ALK2, "synthid_row": "S1"},
    {"id": "alk2_original", "file": "alk2_original.jpg", "title": "The same figure, untouched",
     "credit": "Figure from PMC10376983 (Biomolecules, CC BY 4.0), unaltered.", "synthid_row": "S0"},
    {"id": "gemini_blot", "file": "gemini_whole_blot.jpeg", "title": "A whole blot made by Gemini",
     "credit": "Generated with Gemini for this project. Still carries Google's content credential."},
]


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
        entry = {k: ex[k] for k in ("id", "title", "credit") if k in ex}
        entry.update({"report": name, "file": ex["file"], "hero": ex.get("hero", False),
                      "synthid": synthid.get(ex.get("synthid_row"), "n/a")})
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
