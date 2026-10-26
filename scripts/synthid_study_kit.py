"""Build the upload set for the hand-run SynthID / C2PA study (watermark gap paper).

Design (fixed in advance, see docs/paper/synthid_protocol.md; no adaptive or iterative querying):
for each Gemini donor image, one openly licensed (CC BY) host figure is assigned, and these
conditions are produced:

  whole_original   the donor exactly as Gemini returned it (positive control)
  whole_jpeg85     the donor re-saved as JPEG quality 85
  panel            the donor replaces one blot panel of the host figure   (~0.5-3% of a full figure)
  panel_in_crop    the same composite cropped to ~4x the panel's area     (~25%)
  row              one row of AI bands pasted across a host blot panel      (~1-5%)
  bands3           three adjacent AI bands pasted into a host blot row      (~0.2-1%)
  band1            one AI band pasted into a host blot row                  (<0.3%)
each composite saved as PNG and as JPEG quality 85, plus every host figure unedited (negative control).

Files get random codes so the person uploading cannot tell the condition. key.csv (the
unblinding key) must not be opened until all verdicts are logged in log.csv.
Synthetic, team-made research stimuli; never to be presented as real data.
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

from btw.bands import background, find_bands
from btw.imgio import normalize_polarity, to_gray
from btw.splice import _match_background, _paste, _scale_patch

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["panel", "row", "bands3", "band1"]


def donors_from(path: Path):
    man = path / "manifest.json"
    if man.exists():
        recs = [r for r in json.loads(man.read_text()) if r.get("file")]
        return [(ROOT / r["file"], r) for r in recs]
    return [(p, {"file": str(p)}) for p in sorted(path.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]


def hosts_from(rng):
    meta = {m["file"]: m for m in json.loads((ROOT / "data/real_figures/metadata.json").read_text())}
    panels = json.loads((ROOT / "data/real_panels/panels.json").read_text())
    by_fig = {}
    for p in panels:
        if p["figure"] in meta and p.get("blot_score", 0) >= .8 and p["w"] >= 80 and p["h"] >= 24:
            fw, fh = Image.open(ROOT / "data/real_figures" / p["figure"]).size
            # Skip failed panel detections where the "panel" is most of the figure.
            if p["w"] * p["h"] <= .35 * fw * fh:
                by_fig.setdefault(p["figure"], []).append(p)
    figs = sorted(by_fig)
    rng.shuffle(figs)
    return [(f, meta[f], by_fig[f]) for f in figs]


def gray_bands(gray, min_w=10, min_h=4):
    """Band-shaped blobs only: wider than tall, not smeared across the panel."""
    H, W = gray.shape
    return [b for b in find_bands(gray) if b.w >= min_w and b.h >= min_h and b.darkness > 15
            and 1.3 <= b.w / max(1, b.h) <= 14 and b.w <= .45 * W and b.h <= .35 * H
            and b.y > 2 and b.y + b.h < H - 2 and b.x > 2 and b.x + b.w < W - 2]


def plausible_row(bands):
    w = np.array([b.w for b in bands], float)
    return len(bands) == 1 or (w.std() / w.mean() < .45)


def rows_of(bands):
    rows = []
    for b in sorted(bands, key=lambda b: b.y + b.h / 2):
        c = b.y + b.h / 2
        for r in rows:
            if abs(r["c"] - c) < max(4, b.h):
                r["bands"].append(b)
                break
        else:
            rows.append({"c": c, "bands": [b]})
    for r in rows:
        r["bands"].sort(key=lambda b: b.x)
    return [r for r in rows if r["bands"]]


def compose(fig: Image.Image, panel: dict, donor: Image.Image, level: str, rng: random.Random):
    """Return (composite RGB, changed-pixel mask over the whole figure) or None."""
    F = np.asarray(fig.convert("RGB")).astype(np.float32)
    x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    crop = F[y:y + h, x:x + w].mean(2)
    g, inverted = normalize_polarity(crop)
    dg, _ = normalize_polarity(to_gray(donor))
    H, W = g.shape
    out = g.copy()
    mask = np.zeros((H, W), np.uint8)
    if level == "panel":
        patch = _scale_patch(dg, W, H)
        patch = _match_background(patch, float(np.median(background(g))))
        out, mask = patch, np.ones((H, W), np.uint8)
    else:
        hb, db = gray_bands(g, 8, 3), gray_bands(dg)
        hrows, drows = rows_of(hb), [r for r in rows_of(db) if len(r["bands"]) >= 3]
        if not hrows or not drows:
            return None
        # Replace k real bands with k AI bands: k = 1, 3, or every band in the row.
        hrow = max(hrows, key=lambda r: len(r["bands"]))
        hbands = hrow["bands"]
        k = {"band1": 1, "bands3": 3, "row": len(hbands)}[level]
        if len(hbands) < k:
            return None
        if not plausible_row(hbands):
            return None
        drows = [r for r in drows if len(r["bands"]) >= k and plausible_row(r["bands"])]
        if not drows:
            return None
        drow = max(drows, key=lambda r: sum(b.darkness for b in r["bands"]))
        hs = rng.randrange(0, len(hbands) - k + 1)
        tgt = hbands[hs:hs + k]
        ds = rng.randrange(0, len(drow["bands"]) - k + 1)
        src = drow["bands"][ds:ds + k]
        # Target box: the real bands plus a margin; the same relative margin on the donor.
        tx0, tx1 = min(b.x for b in tgt), max(b.x + b.w for b in tgt)
        ty0, ty1 = min(b.y for b in tgt), max(b.y + b.h for b in tgt)
        mx, my = max(2, int(.15 * (tx1 - tx0) / k)), max(2, int(.8 * (ty1 - ty0)))
        bx0, bx1, by0, by1 = max(0, tx0 - mx), min(W, tx1 + mx), max(0, ty0 - my), min(H, ty1 + my)
        sx0, sx1 = min(b.x for b in src), max(b.x + b.w for b in src)
        sy0, sy1 = min(b.y for b in src), max(b.y + b.h for b in src)
        fx, fy = (sx1 - sx0) / max(1, tx1 - tx0), (sy1 - sy0) / max(1, ty1 - ty0)
        dx0, dx1 = max(0, int(sx0 - (tx0 - bx0) * fx)), min(dg.shape[1], int(sx1 + (bx1 - tx1) * fx))
        dy0, dy1 = max(0, int(sy0 - (ty0 - by0) * fy)), min(dg.shape[0], int(sy1 + (by1 - ty1) * fy))
        if bx1 - bx0 < 6 or by1 - by0 < 4 or dx1 - dx0 < 4 or dy1 - dy0 < 3:
            return None
        # Remove the real bands first (inpaint), so the AI bands replace rather than overlay them.
        g8 = np.clip(g, 0, 255).astype(np.uint8)
        hole = np.zeros_like(g8)
        hole[by0:by1, bx0:bx1] = 1
        base = cv2.inpaint(g8, hole, 3, cv2.INPAINT_TELEA).astype(np.float32)
        patch = _scale_patch(dg[dy0:dy1, dx0:dx1], bx1 - bx0, by1 - by0)
        patch = _match_background(patch, float(np.median(background(g))))
        res, _ = _paste(base, patch, (bx0 + bx1) // 2, (by0 + by1) // 2, "feather", rng)
        if res is None:
            return None
        out = res
        mask = (np.abs(out - g) > 2).astype(np.uint8)
        # Reject implausible pastes: the AI box should be about as dark as the real one.
        real = np.clip(np.median(background(g)) - g[by0:by1, bx0:bx1], 0, None).mean()
        fake = np.clip(np.median(background(g)) - out[by0:by1, bx0:bx1], 0, None).mean()
        if not (.4 <= (fake + 1) / (real + 1) <= 2.5):
            return None
    if mask.sum() < 12:
        return None
    if inverted:
        out = 255.0 - out
    G = F.copy()
    region = G[y:y + h, x:x + w]
    sel = mask.astype(bool)
    for c in range(3):
        region[..., c][sel] = out[sel]
    full = np.zeros(F.shape[:2], np.uint8)
    full[y:y + h, x:x + w] = mask
    return Image.fromarray(np.clip(G, 0, 255).astype(np.uint8)), full


def contact_sheet(out: Path, key: list[dict]):
    """One QC image per host: unedited, panel, row, bands3, band1 cropped around the edits.
    Reviewed BEFORE any upload; a host whose composites look implausible is dropped whole."""
    qc = out / "qc"
    qc.mkdir(exist_ok=True)
    order = ["host_unedited"] + LEVELS
    for host in sorted({r["host"] for r in key if r["host"]}):
        rows = sorted([r for r in key if r["host"] == host and r["format"] == "png" and r["condition"] in order],
                      key=lambda r: order.index(r["condition"]))
        ims = [Image.open(out / "uploads" / r["file"]).convert("RGB") for r in rows]
        a = np.asarray(ims[0]).astype(int)
        d = np.any([np.abs(np.asarray(i).astype(int) - a).sum(2) > 0 for i in ims[1:]], 0)
        ys, xs = np.nonzero(d)
        box = (max(0, xs.min() - 30), max(0, ys.min() - 30), xs.max() + 30, ys.max() + 30)
        tiles = [i.crop(box) for i in ims]
        tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
        sheet = Image.new("RGB", (len(tiles) * (tw + 10), th), "white")
        for j, t in enumerate(tiles):
            sheet.paste(t, (j * (tw + 10), 0))
        name = host.split()[0]
        sheet.save(qc / f"{name}.png")


def save(img: Image.Image, path: Path, fmt: str):
    if fmt == "png":
        img.save(path, "PNG")
    else:
        img.convert("RGB").save(path, "JPEG", quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", default="data/ai_donors_v6/gemini")
    ap.add_argument("--max-donors", type=int, default=6)
    ap.add_argument("--out", default="synthid_kit")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    donors = donors_from(ROOT / a.donors)[: a.max_donors]
    if not donors:
        raise SystemExit(f"No donor images in {a.donors}")
    hosts = hosts_from(rng)
    out = ROOT / a.out
    up = out / "uploads"
    if out.exists():
        shutil.rmtree(out)
    up.mkdir(parents=True)
    key, used = [], set()

    def code():
        while True:
            c = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(5))
            if c not in used:
                used.add(c)
                return c

    def add(img_or_bytes, ext, row):
        c = code()
        p = up / f"{c}.{ext}"
        if isinstance(img_or_bytes, bytes):
            p.write_bytes(img_or_bytes)
        else:
            save(img_or_bytes, p, "png" if ext == "png" else "jpg")
        key.append({"code": c, "file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **row})

    hi = 0
    for dpath, drec in donors:
        donor = Image.open(dpath)
        donor.load()
        add(dpath.read_bytes(), dpath.suffix.lstrip(".").lower(),
            {"condition": "whole_original", "donor": str(drec.get("file")), "host": "", "ai_fraction": 1.0, "format": "as generated"})
        add(donor, "jpg", {"condition": "whole_jpeg85", "donor": str(drec.get("file")), "host": "", "ai_fraction": 1.0, "format": "jpeg85"})
        made = None
        while hi < len(hosts) and made is None:
            fname, meta, panels = hosts[hi]
            hi += 1
            fig = Image.open(ROOT / "data/real_figures" / fname).convert("RGB")
            panel = max(panels, key=lambda p: p["w"] * p["h"])
            comps = {}
            for lv in LEVELS:
                r = None
                for _ in range(8):
                    r = compose(fig, panel, donor, lv, random.Random(rng.random()))
                    if r is not None:
                        break
                if r is None:
                    break
                comps[lv] = r
            if len(comps) == len(LEVELS):
                made = (fname, meta, fig, comps, panel)
        if made is None:
            print("no usable host left for", dpath.name)
            continue
        fname, meta, fig, comps, panel = made
        cite = f"{meta.get('pmcid')} {meta.get('doi')} ({meta.get('license')})"
        add(fig, "png", {"condition": "host_unedited", "donor": "", "host": cite, "ai_fraction": 0.0, "format": "png"})
        for lv, (img, m) in comps.items():
            frac = float(m.mean())
            for fmt in ("png", "jpg"):
                add(img, fmt, {"condition": lv, "donor": str(drec.get("file")), "host": cite,
                               "ai_fraction": round(frac, 5), "format": "png" if fmt == "png" else "jpeg85"})
        # The same AI panel inside a figure crop about 4x its area (~25% AI), the scale at which
        # a whole AI blot placed in a 2x2 figure was detected in the pilot; plus the unedited crop.
        px, py, pw, ph = panel["x"], panel["y"], panel["w"], panel["h"]
        Wf, Hf = fig.size
        cx0, cy0 = max(0, px - pw // 2), max(0, py - ph // 2)
        box = (cx0, cy0, min(Wf, cx0 + 2 * pw), min(Hf, cy0 + 2 * ph))
        img, m = comps["panel"]
        frac = float(m[box[1]:box[3], box[0]:box[2]].mean())
        add(fig.crop(box), "png", {"condition": "host_crop_unedited", "donor": "", "host": cite, "ai_fraction": 0.0, "format": "png"})
        for fmt in ("png", "jpg"):
            add(img.crop(box), fmt, {"condition": "panel_in_crop", "donor": str(drec.get("file")), "host": cite,
                                     "ai_fraction": round(frac, 5), "format": "png" if fmt == "png" else "jpeg85"})
    rng.shuffle(key)
    with open(out / "key.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        wr.writeheader()
        wr.writerows(key)
    contact_sheet(out, key)
    with open(out / "log.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["code", "synthid_verdict_verbatim", "date", "uploader", "notes"])
        for c in sorted(r["code"] for r in key):
            wr.writerow([c, "", "", "", ""])
    print(f"{len(key)} files in {up}; key.csv is the unblinding key, fill log.csv")


if __name__ == "__main__":
    main()
