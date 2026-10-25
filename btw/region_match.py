"""Find the same image region in two figures, even after cropping, resizing,
re-saving, rotation or a mirror flip.

Used two ways:
  * within one upload (intrapaper duplication): every figure against every other
    figure, e.g. one loading control strip reused under two experiments;
  * against a reference library of published figures (interpaper duplication),
    e.g. a blot panel lifted from an earlier paper.

Method: SIFT keypoints on image content only (text on a white background is
masked out, because labels like "beta-actin" are rendered in the same fonts in
every paper), Lowe's ratio test, then a RANSAC similarity transform. A match is
reported only when many keypoints agree on one geometric transform over a
region of real size. This finds copied pixels, not similar looking experiments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = ROOT / "data/real_figures"
CACHE = ROOT / ".cache/reference_index.npz"

# Papers whose figures play the role of "new uploads" in our demos. They are
# left out of the reference library so each demo shows only what it was built
# to show (otherwise every demo figure would trivially match itself).
DEMO_PAPERS = {"PMC10352194", "PMC10755782", "PMC10315040", "PMC10376983"}

MAX_FEATURES = 1500
UPSCALE = 1.0             # SIFT already doubles the image internally
RATIO = 0.75
MIN_INLIERS = 14          # keypoints agreeing on one transform
MIN_SIDE = 24             # px, smallest side of the matched region in both images
MIN_WIDTH = 45            # px; narrow vertical matches are ladder and label columns, not panels

_sift = None


def _detector():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=0.02)
    return _sift


def content_mask(gray: np.ndarray) -> np.ndarray:
    """255 where the figure shows an image (blot, micrograph, photo), 0 on plain
    text and white background. Glyphs are small isolated components; image
    regions are large, filled ones once nearby dark pixels are merged."""
    ink = (gray < 235).astype(np.uint8)
    merged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, 8)
    mask = np.zeros_like(gray, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < 30 or h < 16 or area < 1500:
            continue
        box = gray[y:y + h, x:x + w]
        filled = area / float(w * h)
        # Text lines merge into thin, sparse components; images are filled or textured.
        if filled < 0.45 and float(box.std()) < 40:
            continue
        mask[lab == i] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


@dataclass
class Features:
    pts: np.ndarray            # (n, 2) keypoint xy in the figure's own pixels
    desc: np.ndarray           # (n, 128) float32
    size: tuple[int, int]      # (width, height)
    gray: np.ndarray | None = None
    flipped_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    flipped_desc: np.ndarray = field(default_factory=lambda: np.zeros((0, 128), np.float32))


def features(image: Image.Image, with_flip: bool = False) -> Features:
    gray0 = np.asarray(image.convert("L"))
    mask = content_mask(gray0)
    gray = cv2.resize(gray0, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC) if UPSCALE != 1 else gray0
    mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) if UPSCALE != 1 else mask
    kp, desc = _detector().detectAndCompute(gray, mask)
    pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / UPSCALE
    desc = desc if desc is not None else np.zeros((0, 128), np.float32)
    f = Features(pts, desc.astype(np.float32), (gray0.shape[1], gray0.shape[0]), gray0)
    if with_flip:
        g2 = np.ascontiguousarray(gray[:, ::-1])
        kp2, d2 = _detector().detectAndCompute(g2, np.ascontiguousarray(mask[:, ::-1]))
        # Kept in the mirrored image's coordinates (a similarity transform cannot
        # model a reflection); match() maps the box back.
        f.flipped_pts = np.array([k.pt for k in kp2], np.float32).reshape(-1, 2) / UPSCALE
        f.flipped_desc = (d2 if d2 is not None else np.zeros((0, 128), np.float32)).astype(np.float32)
    return f


def _ratio_matches(qd: np.ndarray, rd: np.ndarray) -> list[tuple[int, int]]:
    """Lowe ratio test, then one-to-one: each reference keypoint keeps only its
    closest query keypoint. Without this, many query points piling onto a few
    bland reference points give RANSAC a degenerate 'transform' to lock onto."""
    if len(qd) < 2 or len(rd) < 2:
        return []
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd, rd, k=2)
    best: dict[int, tuple[float, int]] = {}
    for x in m:
        if len(x) == 2 and x[0].distance < RATIO * x[1].distance:
            a = x[0]
            if a.trainIdx not in best or a.distance < best[a.trainIdx][0]:
                best[a.trainIdx] = (a.distance, a.queryIdx)
    return [(qi, ti) for ti, (_, qi) in best.items()]


def _verify(qp: np.ndarray, rp: np.ndarray, pairs: list[tuple[int, int]]):
    """RANSAC similarity transform; returns (inliers, query box, ref box, scale) or None."""
    if len(pairs) < MIN_INLIERS:
        return None
    src = np.float32([qp[i] for i, _ in pairs])
    dst = np.float32([rp[j] for _, j in pairs])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=4000, confidence=0.995)
    if M is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if n < MIN_INLIERS or not (0.25 < scale < 4.0):
        return None
    qs, rs = src[inl], dst[inl]
    qbox = [float(qs[:, 0].min()), float(qs[:, 1].min()), float(qs[:, 0].max()), float(qs[:, 1].max())]
    rbox = [float(rs[:, 0].min()), float(rs[:, 1].min()), float(rs[:, 0].max()), float(rs[:, 1].max())]
    for b in (qbox, rbox):
        if b[2] - b[0] < max(MIN_SIDE, MIN_WIDTH) or b[3] - b[1] < MIN_SIDE * 0.6:
            return None
    # Repetitive structure (lanes of one blot) can line up keypoints along a
    # single row; require the agreeing points to spread in two dimensions.
    if float(np.std(qs[:, 0])) < 4.0 or float(np.std(qs[:, 1])) < 1.5:
        return None
    return n, qbox, rbox, scale, M


MIN_AGREEMENT = 0.90   # content pixels must agree once aligned; same-template charts reached 0.87, real copies 0.99


def pixel_agreement(qg: np.ndarray, rg: np.ndarray, M: np.ndarray, rbox) -> float:
    """Correlation of the reference region with the query warped onto it.
    Keypoints can line up on a shared chart template or panel layout while the
    content differs; copied pixels also agree pixel by pixel."""
    h, w = rg.shape
    warped = cv2.warpAffine(qg.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(qg, np.float32), M, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    x0, y0, x1, y1 = [int(round(v)) for v in rbox]
    sl = (slice(max(0, y0), min(h, y1 + 1)), slice(max(0, x0), min(w, x1 + 1)))
    a = cv2.GaussianBlur(rg.astype(np.float32), (3, 3), 0)[sl]
    b = cv2.GaussianBlur(warped, (3, 3), 0)[sl]
    # Compare image content only: shared white space, frames and labels would
    # make two different panels in one layout look alike.
    v = valid[sl] & (content_mask(rg)[sl] > 0) & ((a < 235) | (b < 235))
    if v.sum() < 200:
        return 0.0
    a, b = a[v], b[v]
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def match(q: Features, r: Features) -> dict | None:
    """Best verified match of figure q inside figure r (either orientation)."""
    best = None
    for flipped, qp, qd in ((False, q.pts, q.desc), (True, q.flipped_pts, q.flipped_desc)):
        if not len(qd):
            continue
        v = _verify(qp, r.pts, _ratio_matches(qd, r.desc))
        if v and q.gray is not None and r.gray is not None:
            qg = q.gray[:, ::-1] if flipped else q.gray
            agree = pixel_agreement(qg, r.gray, v[4], v[2])
            if agree < MIN_AGREEMENT:
                v = None
        if v and (best is None or v[0] > best["inliers"]):
            qb = v[1]
            if flipped:
                W = q.size[0]
                qb = [W - 1 - qb[2], qb[1], W - 1 - qb[0], qb[3]]
            best = {"inliers": v[0], "query_box": qb, "ref_box": v[2], "scale": round(v[3], 3), "flipped": flipped}
    return best


def _pad(box, size, frac=0.08):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * frac + 4, (y1 - y0) * frac + 4
    return [max(0, int(x0 - px)), max(0, int(y0 - py)), min(size[0], int(x1 + px)), min(size[1], int(y1 + py))]


def within_upload(images: list[Image.Image], feats: list[Features] | None = None, max_figures: int = 30) -> list[dict]:
    """Regions shared between different figures of one upload."""
    images = images[:max_figures]
    feats = feats[:max_figures] if feats else [features(im, with_flip=True) for im in images]
    out = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            m = match(feats[j], feats[i])
            if m:
                out.append({"a": {"figure": i, "box": _pad(m["ref_box"], feats[i].size)},
                            "b": {"figure": j, "box": _pad(m["query_box"], feats[j].size)},
                            "inliers": m["inliers"], "scale": m["scale"], "flipped": m["flipped"]})
    return out


# ---------------------------------------------------------------- library ----

@dataclass
class Library:
    entries: list[dict]        # metadata per figure
    pts: list[np.ndarray]
    desc: list[np.ndarray]
    sizes: list[tuple[int, int]]
    all_desc: np.ndarray       # stacked descriptors for one nearest-neighbour index
    owner: np.ndarray          # figure index of each stacked descriptor
    flann: object = None


def _library_files() -> list[tuple[Path, dict]]:
    meta = {m["file"]: m for m in json.loads((LIBRARY_DIR / "metadata.json").read_text())}
    files = sorted(p for p in LIBRARY_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [(p, meta.get(p.name, {"file": p.name})) for p in files
            if meta.get(p.name, {}).get("pmcid", p.name.split("_")[0]) not in DEMO_PAPERS]


def _fingerprint(files) -> str:
    h = hashlib.sha256()
    for p, _ in files:
        h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
    h.update(f"{MAX_FEATURES}|{sorted(DEMO_PAPERS)}|v1".encode())
    return h.hexdigest()[:16]


_library: Library | None = None


def library() -> Library:
    """The reference library of published figures, cached on disk after the first build."""
    global _library
    if _library is not None:
        return _library
    files = _library_files()
    fp = _fingerprint(files)
    entries, pts, desc, sizes = [], [], [], []
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["fingerprint"]) == fp:
            entries = json.loads(str(z["entries"]))
            counts = z["counts"]
            P, D = z["pts"], z["desc"].astype(np.float32)
            off = 0
            for c in counts:
                pts.append(P[off:off + c]); desc.append(D[off:off + c]); off += c
            sizes = [tuple(s) for s in z["sizes"]]
    if not entries:
        for p, m in files:
            f = features(Image.open(p))
            entries.append({k: m.get(k) for k in ("file", "pmcid", "doi", "year", "title", "figLabel", "license")})
            pts.append(f.pts); desc.append(f.desc); sizes.append(f.size)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, fingerprint=fp, entries=json.dumps(entries),
                            counts=np.array([len(d) for d in desc]),
                            pts=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
                            desc=np.clip(np.concatenate(desc), 0, 255).astype(np.uint8) if desc else np.zeros((0, 128), np.uint8),
                            sizes=np.array(sizes))
        desc = [np.clip(d, 0, 255).astype(np.uint8).astype(np.float32) for d in desc]
    owner = np.concatenate([np.full(len(d), i) for i, d in enumerate(desc)]) if desc else np.zeros(0, int)
    all_desc = np.concatenate(desc).astype(np.float32) if desc else np.zeros((0, 128), np.float32)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    flann.add([all_desc]); flann.train()
    _library = Library(entries, pts, desc, sizes, all_desc, owner, flann)
    return _library


def against_library(images: list[Image.Image], feats: list[Features] | None = None, top: int = 4) -> list[dict]:
    """Regions of the uploaded figures that also appear in a published library figure."""
    lib = library()
    if not len(lib.all_desc):
        return []
    out = []
    for fi, im in enumerate(images):
        q = feats[fi] if feats else features(im, with_flip=True)
        votes = np.zeros(len(lib.entries))
        for qd in (q.desc, q.flipped_desc):
            if len(qd) < 2:
                continue
            for a, b in (x for x in lib.flann.knnMatch(qd, k=2) if len(x) == 2):
                if a.distance < RATIO * b.distance:
                    votes[lib.owner[a.trainIdx]] += 1
        for ri in np.argsort(-votes)[:top]:
            if votes[ri] < MIN_INLIERS:
                break
            gray = np.asarray(Image.open(LIBRARY_DIR / lib.entries[ri]["file"]).convert("L"))
            r = Features(lib.pts[ri], lib.desc[ri], lib.sizes[ri], gray)
            m = match(q, r)
            if m:
                e = lib.entries[ri]
                out.append({"figure": fi, "box": _pad(m["query_box"], q.size),
                            "source": e, "source_box": _pad(m["ref_box"], r.size),
                            "inliers": m["inliers"], "scale": m["scale"], "flipped": m["flipped"]})
    return out


def library_size() -> dict:
    lib = library()
    return {"figures": len(lib.entries), "papers": len({e.get("pmcid") for e in lib.entries})}
