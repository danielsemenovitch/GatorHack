"""How often does the region matcher find a panel lifted into another figure?

For each trial: take a blot or image panel from one published CC BY figure,
transform it the way a reused panel gets transformed (resize, JPEG, sometimes a
mirror flip, sometimes a tighter crop), paste it over a panel of the same size
in a figure from a DIFFERENT paper, and ask the matcher whether the composite
shares a region with the source figure. False matches are measured separately
on every pair of untouched library figures (no edits): see
results/region_match_eval.json.

Run:  python -m scripts.eval_region_match [--trials 120]
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from btw import region_match as rm
from btw.panels import find_panels

ROOT = Path(__file__).resolve().parents[1]


def jpeg(im, q):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--fp", action="store_true", help="also re-run the all-pairs false match check (slow)")
    ap.add_argument("--blocks", action="store_true",
                    help="lift larger regions (a multi-strip blot or a micrograph block, 25 to 55%% of the figure) instead of single panels")
    args = ap.parse_args()
    rng = random.Random(7)
    meta = {m["file"]: m for m in json.loads((rm.LIBRARY_DIR / "metadata.json").read_text())}
    files = sorted(p for p in rm.LIBRARY_DIR.glob("*.jpg"))
    figs = {p.name: Image.open(p).convert("RGB") for p in files}
    panels = {n: [p for p in find_panels(im, min_score=0.0) if p.w >= 60 and p.h >= 24] for n, im in figs.items()}
    names = [n for n in figs if panels[n]]

    rows = []
    for t in range(args.trials):
        src, host = rng.sample(names, 2)
        if meta[src]["pmcid"] == meta[host]["pmcid"]:
            continue
        sp = rng.choice(panels[src]); hp = rng.choice(panels[host])
        crop = figs[src].crop((sp.x, sp.y, sp.x + sp.w, sp.y + sp.h))
        if args.blocks:
            # A larger reused region: a block of the figure that is mostly image
            # content (blots or micrographs), like a whole panel with several strips.
            W, H = figs[src].size
            gray = np.asarray(figs[src].convert("L")); mask = rm.content_mask(gray)
            for _ in range(30):
                bw, bh = int(W * rng.uniform(0.25, 0.55)), int(H * rng.uniform(0.25, 0.55))
                bx, by = rng.randint(0, W - bw), rng.randint(0, H - bh)
                if mask[by:by + bh, bx:bx + bw].mean() / 255 > 0.5:
                    break
            else:
                continue
            crop = figs[src].crop((bx, by, bx + bw, by + bh))
        kind = rng.choice(["resize", "resize+jpeg", "flip", "crop+resize"])
        if kind == "crop+resize":
            cw, ch = int(crop.width * 0.75), int(crop.height * 0.85)
            crop = crop.crop((0, 0, max(30, cw), max(16, ch)))
        if kind == "flip":
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        # Uniform resize (aspect kept), pasted at the host panel's position.
        s = rng.uniform(0.7, 1.3)
        crop = crop.resize((max(20, round(crop.width * s)), max(10, round(crop.height * s))), Image.LANCZOS)
        if args.blocks:
            # Place the whole block inside the host figure (a block hanging off the
            # edge would only be partly reused).
            hw, hh = figs[host].size
            px, py = rng.randint(0, max(0, hw - crop.width)), rng.randint(0, max(0, hh - crop.height))
        else:
            px, py = hp.x, hp.y
        comp = figs[host].copy(); comp.paste(crop, (px, py))
        q = rng.choice([90, 75, 60]) if kind == "resize+jpeg" else 92
        comp = jpeg(comp, q)
        m = rm.match(rm.features(comp, with_flip=True), rm.features(figs[src]))
        rows.append({"source": src, "host": host, "kind": kind, "jpeg": q,
                     "scale": round(s, 2), "found": bool(m), "inliers": m["inliers"] if m else 0})

    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["found"])
    out = {"trials": len(rows), "found": sum(r["found"] for r in rows),
           "by_kind": {k: {"trials": len(v), "found": sum(v)} for k, v in by.items()},
           "note": ("Lifted blocks (25 to 55% of the source figure, mostly blot or micrograph content)" if args.blocks else
                    "Lifted single panels (often one blot strip, at least 60x24 px)") + " pasted into a figure from a different paper.",
           "rows": rows}
    prev = ROOT / ("results/region_match_eval_blocks.json" if args.blocks else "results/region_match_eval.json")
    if args.fp:
        feats = [rm.features(figs[p.name], with_flip=True) for p in files]
        meta_p = {m["file"]: m["pmcid"] for m in json.loads((rm.LIBRARY_DIR / "metadata.json").read_text())}
        found = [(files[i].name, files[j].name) for i, j in itertools.combinations(range(len(files)), 2) if rm.match(feats[j], feats[i])]
        out["untouched_pairs"] = {"pairs": len(files) * (len(files) - 1) // 2, "matches": len(found),
                                  "cross_paper": sum(meta_p[a] != meta_p[b] for a, b in found),
                                  "matched": found, "note": "every pair of the untouched library figures; a same-paper match can be real shared content"}
    elif prev.exists():
        old = json.loads(prev.read_text())
        if "untouched_pairs" in old:
            out["untouched_pairs"] = old["untouched_pairs"]
    prev.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
