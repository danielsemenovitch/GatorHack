"""Build the version 2 SynthID upload set: area, processing and realistic edits kept apart.

Three arms (docs/paper/synthid_protocol.md, amendment 2):

  A. Dose series, raw pixels. For each of the first N Gemini donors (manifest order), an
     untouched crop of the donor centred on its bands (no resizing, no colour change) is pasted into the
     centre of a CC BY host figure so that it covers 50, 20, 10, 5, 2, 1 and 0.5% of the
     uploaded image. Plus the donor itself, untouched and re-saved as JPEG 85.
  B. Processing controls at 100% AI. The whole donor (i) converted to grayscale only, and
     (ii) put through the same steps the realistic band edits use: grayscale, downscale to
     420 px width, darkness map, re-rendered on a flat membrane background.
  C. Realistic edits: the 6 QC-passed donor/host pairs from the version 1 kit, PNG only
     (panel in a crop, panel, row, three bands, one band), plus their unedited hosts.

All arm A and B files are lossless PNG except the JPEG 85 control. Files get fresh random
codes; the key is sealed until all verdicts are logged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005]


def raw_paste(host: Image.Image, donor: Image.Image, frac: float, centre=None):
    """Paste an untouched centre crop of the donor covering `frac` of the host's pixels."""
    H = np.asarray(host.convert("RGB")).copy()
    D = np.asarray(donor.convert("RGB"))
    hh, hw = H.shape[:2]
    dh, dw = D.shape[:2]
    area = frac * hh * hw
    aspect = dw / dh
    cw = int(round(np.sqrt(area * aspect)))
    ch = int(round(area / max(cw, 1)))
    cw, ch = min(cw, dw, hw), min(ch, dh, hh)
    if cw * ch < .9 * area:  # host too wide or tall for the donor aspect: fit the other way
        ch = min(dh, hh)
        cw = min(dw, hw, int(round(area / ch)))
    # Centre the crop on the donor's bands (darkness-weighted), so small crops contain bands,
    # as a real edit would, rather than empty membrane.
    from btw.bands import darkness_map
    from btw.imgio import to_gray
    dk = darkness_map(to_gray(donor))
    thr = np.percentile(dk, 97)
    ys, xs = np.nonzero(dk >= thr)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (dh // 2, dw // 2)
    y0 = int(np.clip(cy - ch // 2, 0, dh - ch))
    x0 = int(np.clip(cx - cw // 2, 0, dw - cw))
    py, px = centre if centre else (hh // 2, hw // 2)
    ty = int(np.clip(py - ch // 2, 0, hh - ch))
    tx = int(np.clip(px - cw // 2, 0, hw - cw))
    H[ty:ty + ch, tx:tx + cw] = D[y0:y0 + ch, x0:x0 + cw]
    return Image.fromarray(H), cw * ch / (hh * hw)


def processed_full(donor: Image.Image) -> Image.Image:
    """The realistic-edit pipeline applied to the whole donor."""
    from btw.bands import background, darkness_map
    from btw.imgio import normalize_polarity, to_gray
    g, _ = normalize_polarity(to_gray(donor))
    g = cv2.resize(g.astype(np.float32), (420, max(8, int(g.shape[0] * 420 / g.shape[1]))), interpolation=cv2.INTER_AREA)
    d = darkness_map(g)
    bg = float(np.median(background(g)))
    out = np.clip(bg - d, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=int, default=10)
    ap.add_argument("--v1", default="synthid_kit")
    ap.add_argument("--out", default="synthid_kit_v2")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    up = out / "uploads"
    up.mkdir(parents=True)
    (out / "screens").mkdir()
    (out / "sealed").mkdir()
    key, used = [], set()

    def add(obj, ext, row):
        while True:
            c = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(5))
            if c not in used:
                used.add(c)
                break
        p = up / f"{c}.{ext}"
        if isinstance(obj, bytes):
            p.write_bytes(obj)
        elif ext == "png":
            obj.save(p, "PNG")
        else:
            obj.convert("RGB").save(p, "JPEG", quality=85)
        key.append({"code": c, "file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "arm": row.pop("arm"), **row})

    # Donors and hosts for arms A and B
    man = sorted([r for r in json.loads((ROOT / "data/ai_donors_v6/gemini/manifest.json").read_text()) if r.get("file")],
                 key=lambda r: r["prompt_index"])[: a.donors]
    meta = json.loads((ROOT / "data/real_figures/metadata.json").read_text())
    v1_hosts = {r["host"].split()[-1] for r in csv.DictReader(open(ROOT / a.v1 / "sealed/key.csv")) if r["host"]}
    # Hosts: Western blot figures only (a detected blot panel), not used in arm C; the paste is
    # centred on the figure's largest blot panel.
    panels = json.loads((ROOT / "data/real_panels/panels.json").read_text())
    blot = {}
    for pnl in panels:
        if pnl.get("blot_score", 0) >= .85:
            if pnl["figure"] not in blot or pnl["w"] * pnl["h"] > blot[pnl["figure"]]["w"] * blot[pnl["figure"]]["h"]:
                blot[pnl["figure"]] = pnl
    hosts = [m for m in meta if m["file"] not in v1_hosts and m["file"] in blot]
    random.Random(7).shuffle(hosts)
    hosts = [m for m in hosts if np.prod(Image.open(ROOT / "data/real_figures" / m["file"]).size) >= 250_000]
    for i, rec in enumerate(man):
        dpath = ROOT / rec["file"]
        donor = Image.open(dpath)
        donor.load()
        model = rec["model"]
        add(dpath.read_bytes(), dpath.suffix.lstrip(".").lower(),
            {"arm": "A", "condition": "whole_original", "donor": rec["file"], "model": model, "host": "", "ai_fraction": 1.0, "format": "as generated"})
        add(donor, "jpg", {"arm": "A", "condition": "whole_jpeg85", "donor": rec["file"], "model": model, "host": "", "ai_fraction": 1.0, "format": "jpeg85"})
        add(donor.convert("L").convert("RGB"), "png",
            {"arm": "B", "condition": "whole_gray", "donor": rec["file"], "model": model, "host": "", "ai_fraction": 1.0, "format": "png"})
        add(processed_full(donor), "png",
            {"arm": "B", "condition": "whole_processed", "donor": rec["file"], "model": model, "host": "", "ai_fraction": 1.0, "format": "png"})
        hm = hosts[i]
        host = Image.open(ROOT / "data/real_figures" / hm["file"]).convert("RGB")
        cite = f"{hm.get('pmcid')} {hm.get('doi')} ({hm.get('license')}) {hm['file']}"
        for f in FRACTIONS:
            pb = blot[hm["file"]]
            img, actual = raw_paste(host, donor, f, centre=(pb["y"] + pb["h"] // 2, pb["x"] + pb["w"] // 2))
            add(img, "png", {"arm": "A", "condition": f"raw_{f:g}", "donor": rec["file"], "model": model, "host": cite,
                             "ai_fraction": round(actual, 5), "format": "png"})
        add(host, "png", {"arm": "A", "condition": "host_unedited", "donor": "", "model": "", "host": cite, "ai_fraction": 0.0, "format": "png"})

    # Arm C: realistic edits from version 1, PNG only, files copied unchanged
    v1 = list(csv.DictReader(open(ROOT / a.v1 / "sealed/key.csv")))
    for r in v1:
        if r["format"] == "jpeg85" or r["condition"] in ("whole_original",):
            continue  # whole originals are already in arm A for overlapping donors; JPEG copies dropped
        if r["condition"] == "whole_jpeg85":
            continue
        src = ROOT / a.v1 / "uploads" / r["file"]
        model = next((m["model"] for m in json.loads((ROOT / "data/ai_donors_v6/gemini/manifest.json").read_text())
                      if m.get("file") == r["donor"]), "")
        add(src.read_bytes(), src.suffix.lstrip("."), {"arm": "C", "condition": r["condition"], "donor": r["donor"], "model": model,
                                                      "host": r["host"], "ai_fraction": r["ai_fraction"], "format": r["format"]})
    # Whole originals for arm C donors not already in arm A
    a_donors = {k["donor"] for k in key if k["arm"] == "A"}
    for d in sorted({r["donor"] for r in v1 if r["donor"]} - a_donors):
        p = ROOT / d
        add(p.read_bytes(), p.suffix.lstrip(".").lower(), {"arm": "C", "condition": "whole_original", "donor": d, "model": "",
                                                          "host": "", "ai_fraction": 1.0, "format": "as generated"})
    rng.shuffle(key)
    with open(out / "sealed/key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "synthid_verdict_verbatim", "date", "uploader", "notes"])
        for c in sorted(k["code"] for k in key):
            w.writerow([c, "", "", "", ""])
    from collections import Counter
    print(len(key), "files;", dict(Counter((k["arm"], k["condition"]) for k in key)))


if __name__ == "__main__":
    main()
