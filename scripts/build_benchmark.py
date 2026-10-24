"""Build the Below the Watermark benchmark.

Steps
  1. unpack the Europe PMC bundle into data/real_figures (+ metadata.json)
  2. cut blot panels out of each figure into data/real_panels
  3. make clean and forged versions of every panel, each pushed through a
     random publishing chain, with pixel masks, into data/benchmark

Splits are by source paper (PMCID), so no paper contributes to both train
and test. Run:  python -m scripts.build_benchmark --bundle path/to/bundle.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from btw.imgio import load_rgb
from btw.panels import find_panels
from btw.publish import random_publish
from btw.splice import METHODS, forge
from btw.sources import ai_panels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def unpack(bundle: Path):
    """Add a Europe PMC bundle to data/real_figures (merging with what is there)."""
    out = DATA / "real_figures"
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "metadata.json"
    meta = {m["file"]: m for m in json.loads(mf.read_text())} if mf.exists() else {}
    items = json.loads(bundle.read_text())
    for it in items:
        (out / it["file"]).write_bytes(base64.b64decode(it.pop("b64")))
        meta[it["file"]] = it
    mf.write_text(json.dumps(list(meta.values()), indent=1))
    print(f"unpacked {len(items)} figures ({len(meta)} total)")


def filter_panels():
    """Keep panels that look like blots: grey, banded, not text heavy, not tiny."""
    import cv2
    from btw.bands import find_bands
    from btw.imgio import to_gray
    pdir = DATA / "real_panels"
    rows = json.loads((pdir / "panels.json").read_text())

    def textiness(g):
        dark = (g < 90).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
        small = sum(1 for i in range(1, n) if 4 <= st[i, 4] <= 250 and st[i, 2] <= 25 and st[i, 3] <= 25)
        return small / (g.size / 10000), float((g > 235).mean())

    keep = []
    for x in rows:
        im = Image.open(pdir / x["panel"]).convert("RGB")
        g = to_gray(im)
        sat = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV)[..., 1].mean() / 255
        t, white = textiness(g)
        if (x["blot_score"] >= 0.72 and sat < 0.10 and len(find_bands(g)) >= 2 and x["w"] >= 60
                and x["h"] >= 20 and not (white > 0.45 and t > 6) and x["w"] * x["h"] < 400000 and t < 25):
            keep.append(x["panel"])
    (pdir / "keep.json").write_text(json.dumps(keep))
    print(f"kept {len(keep)} of {len(rows)} panels from {len({x['pmcid'] for x in rows if x['panel'] in set(keep)})} papers")


def extract_panels(min_score: float = 0.35):
    src = DATA / "real_figures"
    out = DATA / "real_panels"
    out.mkdir(parents=True, exist_ok=True)
    meta = {m["file"]: m for m in json.loads((src / "metadata.json").read_text())}
    rows = []
    for f in sorted(src.glob("*.jpg")):
        im = load_rgb(f)
        for p in find_panels(im, min_score=min_score):
            if p.w < 60 or p.h < 20:
                continue
            name = f"{f.stem}_p{p.id}.png"
            p.crop(im).save(out / name)
            rows.append({"panel": name, "figure": f.name, "pmcid": meta.get(f.name, {}).get("pmcid"),
                         **p.to_dict()})
    (out / "panels.json").write_text(json.dumps(rows, indent=1))
    print(f"extracted {len(rows)} panels")


def split_of(pmcid: str) -> str:
    h = int(hashlib.md5(pmcid.encode()).hexdigest(), 16) % 100
    return "test" if h < 30 else "train"


def build(n_clean: int = 2, n_forged_per_method: int = 2, seed: int = 0):
    rng = random.Random(seed)
    pdir = DATA / "real_panels"
    panels = json.loads((pdir / "panels.json").read_text())
    keep_file = pdir / "keep.json"
    if keep_file.exists():  # optional manual QA list
        keep = set(json.loads(keep_file.read_text()))
        panels = [p for p in panels if p["panel"] in keep]
    imgs = {p["panel"]: load_rgb(pdir / p["panel"]) for p in panels}
    ai = ai_panels()
    out = DATA / "benchmark"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    index = []
    for p in panels:
        split = split_of(p["pmcid"] or p["figure"])
        same_split_donors = [imgs[q["panel"]] for q in panels
                             if q["pmcid"] != p["pmcid"] and split_of(q["pmcid"] or q["figure"]) == split]
        base = imgs[p["panel"]]
        jobs = [("clean", None)] * n_clean + [(m, m) for m in METHODS for _ in range(n_forged_per_method)]
        for k, (kind, method) in enumerate(jobs):
            if method is None:
                img, mask, meta = base, np.zeros(base.size[::-1], np.uint8), {"method": "clean", "blend": ""}
            else:
                fg = forge(base, method, rng, donor_panels=same_split_donors, ai_panels=ai)
                if fg is None:
                    continue
                img, mask, meta = fg.image, fg.mask, {"method": fg.method, "blend": fg.blend}
            pub, steps = random_publish(img, rng)
            if pub.size != img.size:
                mask = np.asarray(Image.fromarray(mask * 255).resize(pub.size, Image.NEAREST)) // 255
            sid = f"{Path(p['panel']).stem}_{k:02d}_{meta['method']}"
            pub.save(out / "images" / f"{sid}.png")
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(out / "masks" / f"{sid}.png")
            index.append({"id": sid, "split": split, "pmcid": p["pmcid"], "panel": p["panel"],
                          "label": int(meta["method"] != "clean"), **meta,
                          "publish": "+".join(steps), "w": pub.size[0], "h": pub.size[1],
                          "forged_px": int((mask > 0).sum())})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    n_f = sum(r["label"] for r in index)
    print(f"benchmark: {len(index)} images ({n_f} forged, {len(index) - n_f} clean)")
    for s in ("train", "test"):
        print(s, sum(r["split"] == s for r in index))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, nargs="*")
    ap.add_argument("--skip-panels", action="store_true")
    ap.add_argument("--n-clean", type=int, default=2)
    ap.add_argument("--n-forged", type=int, default=2)
    a = ap.parse_args()
    for b in a.bundle or []:
        unpack(b)
    if not a.skip_panels:
        extract_panels()
        filter_panels()
    build(a.n_clean, a.n_forged)
