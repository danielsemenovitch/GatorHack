"""Turn an uploaded image or PDF into the report the web app renders."""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .detector import Detector
from .duplicates import find_duplicates
from .imgio import load_rgb, to_gray
from .panels import find_panels
from .provenance import read_c2pa

_DETECTOR: Detector | None = None


def detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        from .detector import load_best
        _DETECTOR = load_best()
    return _DETECTOR


def _data_url(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _heat_overlay(heat: np.ndarray, threshold: float) -> Image.Image:
    """Transparent PNG: only regions above the flag line are tinted (Ponceau pink)."""
    h = cv2.GaussianBlur(np.clip(heat, 0, 1).astype(np.float32), (0, 0), 3.0)
    a = np.clip((h - threshold) / max(1e-3, 1 - threshold), 0, 1) ** 0.7
    rgba = np.zeros((*h.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 200, 40, 95
    rgba[..., 3] = (a * 190).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _figures_from_upload(data: bytes, filename: str) -> list[tuple[str, Image.Image]]:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf as fitz
        doc = fitz.open(stream=data, filetype="pdf")
        out = []
        for pno, page in enumerate(doc):
            for k, img in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, img[0])
                if pix.width < 150 or pix.height < 80:
                    continue
                if pix.n - pix.alpha > 3 or pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                out.append((f"page {pno + 1}, image {k + 1}", im))
        return out
    return [("uploaded image", load_rgb(data))]


def analyze(data: bytes, filename: str = "upload", max_panels: int = 24) -> dict[str, Any]:
    t0 = time.time()
    det = detector()
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    provenance = read_c2pa(data, mime) if not filename.lower().endswith(".pdf") else {
        "present": False, "summary": "PDF upload: credentials are checked per image only for image uploads."}
    figures = _figures_from_upload(data, filename)
    report_figs, all_panels_gray, panel_refs = [], [], []
    n_panels = 0
    for fi, (label, fig) in enumerate(figures):
        panels = find_panels(fig)
        fig_entry = {"id": fi, "label": label, "width": fig.width, "height": fig.height,
                     "image": _data_url(fig if max(fig.size) <= 1600 else fig.resize(
                         (int(fig.width * 1600 / max(fig.size)), int(fig.height * 1600 / max(fig.size)))), "JPEG"),
                     "scale": min(1.0, 1600 / max(fig.size)), "panels": []}
        for p in panels:
            if n_panels >= max_panels:
                break
            crop = p.crop(fig)
            res = det.analyze(crop)
            all_panels_gray.append(to_gray(crop))
            panel_refs.append((fi, p.id))
            fig_entry["panels"].append({
                **p.to_dict(), "global_index": n_panels,
                "score": round(res.score, 3), "flagged": bool(res.score > det.threshold),
                "bands": res.bands, "heat": _data_url(_heat_overlay(res.heat, det.threshold)),
                "crop": _data_url(crop), "why": [{"feature": f, "value": round(v, 2)} for f, v in res.top_features],
            })
            n_panels += 1
        report_figs.append(fig_entry)
    dups = find_duplicates(all_panels_gray) if all_panels_gray else []
    for d in dups:
        for side in ("a", "b"):
            fi, pid = panel_refs[d[side]["panel"]]
            d[side]["figure"], d[side]["panel_id"] = fi, pid
    flagged = [p for f in report_figs for p in f["panels"] if p["flagged"]]
    top = max((p["score"] for f in report_figs for p in f["panels"]), default=0.0)
    verdict = ("suspicious" if flagged or dups else ("no_blots" if n_panels == 0 else "no_findings"))
    return {
        "filename": filename,
        "verdict": verdict,
        "summary": {
            "figures": len(report_figs), "panels": n_panels, "flagged_panels": len(flagged),
            "duplicate_pairs": len(dups), "max_splice_score": round(top, 3),
            "threshold": round(det.threshold, 3),
            "content_credentials": provenance.get("present", False),
        },
        "provenance": provenance,
        "synthid": {
            "status": "manual",
            "note": ("SynthID Detector has no public API and its terms forbid automated "
                     "submissions, so download a panel and check it at synthid.com yourself. In our "
                     "tests it caught whole AI images every time, while single bands were too "
                     "small for it to check."),
            "url": "https://synthid.com",
        },
        "figures": report_figs,
        "duplicates": dups,
        "seconds": round(time.time() - t0, 2),
    }
