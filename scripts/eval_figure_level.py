"""Whole-figure, end-to-end evaluation of the app pipeline (routing + CNN v3 + flag line).

For every figure from a held-out TEST paper (never used to train v3) that has at
least one automatically routed blot panel, we paste one Gemini lane into one
routed panel (btw.splice.forge, "ai_splice"), put the panel back into the
figure, save the figure as JPEG quality 90, and run report.analyze on both the
original and the edited figure, exactly as an upload would be analyzed.

Reports, at the research threshold (the app default) and the strict line:
  * how often the edited panel is flagged,
  * how often at least one UNTOUCHED panel is flagged (edited figure),
  * how often the untouched original figure gets any flag.
Edited figures where the edited panel is flagged and nothing else is are saved
to results/figure_level_examples/ (with masks) as candidate demo examples; the
report says how many trials they were selected from.

Run:  python -m scripts.eval_figure_level
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from btw import report
from btw.sources import ai_panels
from btw.splice import forge
from scripts.build_benchmark import split_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figure_level_examples"


def jpeg_bytes(im: Image.Image, q: int = 90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


def overlaps(p, box):
    return p["x"] < box[2] and p["x"] + p["w"] > box[0] and p["y"] < box[3] and p["y"] + p["h"] > box[1]


def main():
    rng = random.Random(2026)
    meta = {m["file"]: m for m in json.loads((ROOT / "data/real_figures/metadata.json").read_text())}
    ai = ai_panels()
    OUT.mkdir(parents=True, exist_ok=True)
    trials = []
    for f in sorted((ROOT / "data/real_figures").glob("*.jpg")):
        m = meta.get(f.name, {})
        if split_of(m.get("pmcid") or f.name) != "test":
            continue
        fig = Image.open(f).convert("RGB")
        cands = report._blot_candidates(fig, "auto")
        if not cands:
            continue
        panel, crop = cands[rng.randrange(len(cands))]
        fg = None
        for _ in range(6):
            fg = forge(crop, "ai_splice", rng, ai_panels=ai)
            if fg is not None:
                break
        if fg is None:
            continue
        edited = fig.copy()
        edited.paste(fg.image, (panel.x, panel.y))
        ys, xs = np.nonzero(fg.mask)
        box = (panel.x + int(xs.min()), panel.y + int(ys.min()), panel.x + int(xs.max()) + 1, panel.y + int(ys.max()) + 1)
        orig_bytes, edit_bytes = jpeg_bytes(fig), jpeg_bytes(edited)
        row = {"figure": f.name, "pmcid": m.get("pmcid"), "edit_box": box, "routed_panels": len(cands)}
        for policy in ("benchmark", "conservative"):
            r_e = report.analyze(edit_bytes, f.name, screening_policy=policy)
            r_o = report.analyze(orig_bytes, f.name, screening_policy=policy)
            pe = r_e["figures"][0]["panels"]
            row[policy] = {
                "edited_panel_flagged": any(p["flagged"] and overlaps(p, box) for p in pe),
                "edited_panel_screened": any(overlaps(p, box) for p in pe),
                "untouched_flagged_in_edited": sum(p["flagged"] and not overlaps(p, box) for p in pe),
                "original_any_flag": r_o["summary"]["flagged_panels"] > 0,
            }
        trials.append(row)
        b = row["benchmark"]
        if b["edited_panel_flagged"] and b["untouched_flagged_in_edited"] == 0 and not b["original_any_flag"]:
            stem = f.stem
            (OUT / f"{stem}_ai_band.jpg").write_bytes(edit_bytes)
            (OUT / f"{stem}_original.jpg").write_bytes(orig_bytes)
            Image.fromarray((np.asarray(fg.mask) > 0).astype(np.uint8) * 255).save(OUT / f"{stem}_mask_in_panel.png")
            row["saved_as_example"] = f"{stem}_ai_band.jpg"
        print(f.name, row["benchmark"], flush=True)
    n = len(trials)
    summ = {}
    for policy in ("benchmark", "conservative"):
        sc = [t for t in trials if t[policy]["edited_panel_screened"]]
        summ[policy] = {
            "figures": n,
            "edited_panel_still_routed": len(sc),
            "edited_panel_flagged": sum(t[policy]["edited_panel_flagged"] for t in trials),
            "edited_figure_with_untouched_panel_flagged": sum(t[policy]["untouched_flagged_in_edited"] > 0 for t in trials),
            "original_figure_with_any_flag": sum(t[policy]["original_any_flag"] for t in trials),
            "clean_hit": sum(t[policy]["edited_panel_flagged"] and t[policy]["untouched_flagged_in_edited"] == 0
                             and not t[policy]["original_any_flag"] for t in trials),
        }
    out = {"description": __doc__.strip().splitlines()[0], "papers": len({t["pmcid"] for t in trials}),
           "summary": summ, "trials": trials,
           "note": "Simulated edits (one Gemini lane per figure) on held-out CC BY test figures; JPEG q90 after editing. "
                   "Every edit was made by us; the published figures are unaltered and their authors are not implicated."}
    (ROOT / "results" / "figure_level_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
