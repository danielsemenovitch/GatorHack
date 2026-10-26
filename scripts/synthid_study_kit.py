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
        recs = sorted([r for r in json.loads(man.read_text()) if r.get("file")], key=lambda r: r.get("prompt_index", 0))
        return [(ROOT / r["file"], r) for r in recs]
    return [(p, {"file": str(p)}) for p in sorted(path.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]


def hosts_from(rng):
    meta = {m["file"]: m for m in json.loads((ROOT / "data/real_figures/metadata.json").read_text())}
    panels = json.loads((ROOT / "data/real_panels/panels.json").read_text())
    by_fig = {}
    for p in panels:
        if p["figure"] in meta and p.get("blot_score", 0) >= .85 and p["w"] >= 80 and p["h"] >= 24:
            fw, fh = Image.open(ROOT / "data/real_figures" / p["figure"]).size
            # Skip failed panel detections where the "panel" is most of the figure.
            if p["w"] * p["h"] <= .35 * fw * fh:
                by_fig.setdefault(p["figure"], []).append(p)
    figs = sorted(by_fig)
    rng.shuffle(figs)
    return [(f, meta[f], by_fig[f]) for f in figs]


def band_rows(gray: np.ndarray, donor: bool):
    """Rows of separate bands as [{"c": y centre, "h": band height, "xs": [x centres], "peak"}].

    Hosts (published panels) use the repository's lane-profile band finder. Donors (clean AI
    images) use connected components of the thresholded darkness map. Rows need >= 3 bands that
    are evenly spaced and not merged into one blob."""
    from btw.bands import darkness_map
    d = darkness_map(gray)
    H, W = gray.shape
    if donor:
        d8 = np.clip(d, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (d8 > max(thr, 20)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        blobs = [(cen[i][0], cen[i][1], st[i][2], st[i][3]) for i in range(1, n)
                 if st[i][4] >= 20 and 1.3 <= st[i][2] / max(1, st[i][3]) <= 15 and st[i][2] <= .3 * W]
    else:
        blobs = [(b.x + b.w / 2, b.y + b.h / 2, b.w, b.h) for b in find_bands(gray) if b.w >= 6 and b.h >= 3]
    rows = []
    for bx, by, bw, bh in sorted(blobs, key=lambda t: t[1]):
        for r in rows:
            if abs(r["c"] - by) < max(3, .6 * bh):
                r["b"].append((bx, by, bw, bh))
                break
        else:
            rows.append({"c": by, "b": [(bx, by, bw, bh)]})
    out = []
    for r in rows:
        b = sorted(r["b"])
        if len(b) < 3:
            continue
        xs = np.array([t[0] for t in b]); ws = np.array([t[2] for t in b]); hs = np.array([t[3] for t in b])
        gaps = np.diff(xs)
        if gaps.std() / gaps.mean() > .35 or ws.max() > 1.8 * np.median(ws) or np.any(gaps < .8 * np.median(ws)):
            continue
        c = float(np.median([t[1] for t in b])); h = float(np.median(hs))
        if c - 1.6 * h < 0 or c + 1.6 * h > H:
            continue
        peak = float(np.median([d[int(t[1]), int(t[0])] for t in b]))
        out.append({"c": c, "h": h, "xs": [float(x) for x in xs], "w": float(np.median(ws)),
                    "pitch": float(np.median(gaps)), "peak": max(peak, 1.0)})
    return out


def warp_strip(dark, xs_src, xs_dst, x0_dst, width, y0_src, y1_src, height):
    """Resample a donor darkness strip so donor band centres land on host lane centres."""
    xs_src, xs_dst = np.asarray(xs_src, float), np.asarray(xs_dst, float)
    dst = np.arange(width, dtype=np.float32) + x0_dst
    mapx = np.interp(dst, xs_dst, xs_src, left=np.nan, right=np.nan)
    # Extrapolate linearly outside the first/last centre with the local scale.
    sl = (xs_src[1] - xs_src[0]) / (xs_dst[1] - xs_dst[0]) if len(xs_src) > 1 else 1.0
    sr = (xs_src[-1] - xs_src[-2]) / (xs_dst[-1] - xs_dst[-2]) if len(xs_src) > 1 else 1.0
    left, right = dst < xs_dst[0], dst > xs_dst[-1]
    mapx[left] = xs_src[0] + (dst[left] - xs_dst[0]) * sl
    mapx[right] = xs_src[-1] + (dst[right] - xs_dst[-1]) * sr
    mapy = np.linspace(y0_src, y1_src, height, dtype=np.float32)
    MX, MY = np.meshgrid(mapx.astype(np.float32), mapy)
    return cv2.remap(dark.astype(np.float32), MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def compose(fig: Image.Image, panel: dict, donor: Image.Image, level: str, rng: random.Random):
    """Return (composite RGB, changed-pixel mask over the whole figure) or None."""
    F = np.asarray(fig.convert("RGB")).astype(np.float32)
    x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    crop = F[y:y + h, x:x + w].mean(2)
    g, inverted = normalize_polarity(crop)
    dg, _ = normalize_polarity(to_gray(donor))
    # Generated images are ~1400 px wide; the band finder is tuned for published panel scale.
    if dg.shape[1] > 480:
        dg = cv2.resize(dg.astype(np.float32), (420, max(8, int(dg.shape[0] * 420 / dg.shape[1]))), interpolation=cv2.INTER_AREA)
    H, W = g.shape
    out = g.copy()
    mask = np.zeros((H, W), np.uint8)
    if level == "panel":
        patch = _scale_patch(dg, W, H)
        patch = _match_background(patch, float(np.median(background(g))))
        out, mask = patch, np.ones((H, W), np.uint8)
    else:
        from btw.bands import darkness_map
        hrows, drows = band_rows(g, False), band_rows(dg, True)
        if not hrows or not drows:
            return None
        hrows = [r for r in hrows if r["peak"] >= 30]  # real, visible bands only
        if not hrows:
            return None
        hrow = hrows[rng.randrange(len(hrows))]
        m = len(hrow["xs"])
        k = m if level == "row" else {"band1": 1, "bands3": 3}[level]
        cand = [r for r in drows if len(r["xs"]) >= k]
        if not cand:
            return None
        drow = cand[rng.randrange(len(cand))]
        j = rng.randrange(0, m - k + 1)
        dj = rng.randrange(0, len(drow["xs"]) - k + 1)
        hx = hrow["xs"][j:j + k]
        dx = drow["xs"][dj:dj + k]
        hp = hrow["pitch"] if k > 1 or m > 1 else hrow["w"] * 1.4
        dp = drow["pitch"]
        # Map donor centres (plus half a lane either side) onto host centres.
        src = [dx[0] - dp / 2] + dx + [dx[-1] + dp / 2]
        dst = [hx[0] - hp / 2] + hx + [hx[-1] + hp / 2]
        bx0, bx1 = int(round(dst[0])), int(round(dst[-1]))
        hh, dh = hrow["h"], drow["h"]
        by0, by1 = int(round(hrow["c"] - 1.6 * hh)), int(round(hrow["c"] + 1.6 * hh))
        bx0, bx1, by0, by1 = max(0, bx0), min(W, bx1), max(0, by0), min(H, by1)
        if bx1 - bx0 < 6 or by1 - by0 < 4:
            return None
        # The box must sit inside one membrane strip: no white separator or frame line in it.
        boxg = g[by0:by1, bx0:bx1]
        lbg = background(g)[by0:by1, bx0:bx1]
        rmeans = boxg.mean(1)
        if (rmeans.max() - np.median(rmeans) > 25 and rmeans.max() > np.median(lbg) + 15) or \
                (boxg.min(1) > np.median(lbg) - 3).mean() > .5:
            return None
        # No white gaps, frame edges or label text inside the box (they mean it straddles strips).
        lvl = float(np.median(lbg))
        if lvl < 235 and (boxg > max(lvl + 18, 238)).mean() > .02:
            return None
        # A row that is dark across nearly the whole box is a frame line or a merged smear.
        if ((boxg < lvl - 40).mean(1) > .85).any():
            return None
        # Host bands split in two by detection: real lanes are separated by clear gaps.
        hb_w = hrow["w"]
        if hrow["pitch"] < 1.25 * hb_w:
            return None
        ddark = darkness_map(dg)
        strip = warp_strip(ddark, src, dst, bx0, bx1 - bx0,
                           drow["c"] - 1.6 * dh, drow["c"] + 1.6 * dh, by1 - by0)
        # Build on the host's own membrane (bands removed by the background estimate), so the
        # pasted lanes keep the local background level and texture instead of a flat box.
        hd = darkness_map(g)
        g8 = np.clip(g, 0, 255).astype(np.uint8)
        hole = ((hd > max(8.0, .2 * hrow["peak"])) & True).astype(np.uint8)
        hole[:by0], hole[by1:], hole[:, :bx0], hole[:, bx1:] = 0, 0, 0, 0
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        base = cv2.inpaint(g8, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)[by0:by1, bx0:bx1]
        patch = np.clip(base - strip * (hrow["peak"] * rng.uniform(.8, 1.2) / drow["peak"]), 0, 255)
        region = g[by0:by1, bx0:bx1]
        ph, pw = patch.shape
        alpha = np.ones_like(patch)
        ky, kx = max(1, ph // 5), max(1, int(hp // 5))
        alpha[:ky] *= np.linspace(0, 1, ky)[:, None]
        alpha[-ky:] *= np.linspace(1, 0, ky)[:, None]
        alpha[:, :kx] *= np.linspace(0, 1, kx)[None, :]
        alpha[:, -kx:] *= np.linspace(1, 0, kx)[None, :]
        out = g.copy()
        out[by0:by1, bx0:bx1] = region * (1 - alpha) + patch * alpha
        mask = (np.abs(out - g) > 2).astype(np.uint8)
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
        name = host.split()[-1].rsplit(".", 1)[0]
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

    used_hosts = set()
    for dpath, drec in donors:
        donor = Image.open(dpath)
        donor.load()
        add(dpath.read_bytes(), dpath.suffix.lstrip(".").lower(),
            {"condition": "whole_original", "donor": str(drec.get("file")), "host": "", "ai_fraction": 1.0, "format": "as generated"})
        add(donor, "jpg", {"condition": "whole_jpeg85", "donor": str(drec.get("file")), "host": "", "ai_fraction": 1.0, "format": "jpeg85"})
        made = None
        for fname, meta, panels in hosts:
            if made is not None:
                break
            if fname in used_hosts:
                continue
            fig = Image.open(ROOT / "data/real_figures" / fname).convert("RGB")
            for panel in sorted(panels, key=lambda p: -p["w"] * p["h"]):
                comps = {}
                for lv in LEVELS:
                    r = None
                    for _ in range(12):
                        r = compose(fig, panel, donor, lv, random.Random(rng.random()))
                        if r is not None:
                            break
                    if r is None:
                        break
                    comps[lv] = r
                if len(comps) == len(LEVELS):
                    break
            if len(comps) == len(LEVELS):
                made = (fname, meta, fig, comps, panel)
                used_hosts.add(fname)
        if made is None:
            print("no usable host left for", dpath.name)
            continue
        fname, meta, fig, comps, panel = made
        cite = f"{meta.get('pmcid')} {meta.get('doi')} ({meta.get('license')}) {fname}"
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
