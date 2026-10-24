"""Inspect raster figures / PDFs with general reuse checks and optional blot screening.

This tool generates leads for HUMAN review, not authenticity or misconduct verdicts.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .bands import find_bands
from .duplicates import find_duplicates
from .generic_duplicates import find_image_reuse
from .imgio import load_rgb, to_gray
from .panels import find_panels
from .provenance import read_c2pa

_DETECTOR = None


def detector():
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
    h = cv2.GaussianBlur(np.clip(heat, 0, 1).astype(np.float32), (0, 0), 3.0)
    a = np.clip((h - threshold) / max(1e-3, 1 - threshold), 0, 1) ** 0.7
    rgba = np.zeros((*h.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 200, 40, 95
    rgba[..., 3] = (a * 190).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _figures_from_upload(data: bytes, filename: str, max_figures: int = 48):
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf as fitz
        doc = fitz.open(stream=data, filetype="pdf")
        out = []
        for pno, page in enumerate(doc):
            for k, img in enumerate(page.get_images(full=True)):
                if len(out) >= max_figures:
                    break
                pix = fitz.Pixmap(doc, img[0])
                if pix.width < 150 or pix.height < 80 or pix.width * pix.height > 12000000:
                    continue
                if pix.n - pix.alpha > 3 or pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                out.append((f"page {pno + 1}, image {k + 1}", im))
            if len(out) >= max_figures:
                break
        doc.close()
        return out
    im = load_rgb(data)
    if im.width * im.height > 12000000:
        raise ValueError("Image exceeds the 12 megapixel analysis limit.")
    return [("uploaded image", im)]


def _blot_candidates(fig: Image.Image):
    """Conservative heuristic gate; never apply CNN to arbitrary extracted images.

    It is NOT a validated general modality classifier. Ambiguous candidates
    are skipped (false negatives are preferable to misleading cross-modality ML).
    """
    result = []
    for panel in find_panels(fig, min_score=0.75):
        if panel.blot_score < 0.82 or panel.w < 90 or panel.h < 24:
            continue
        if panel.w / max(1, panel.h) < 1.5:
            continue
        crop = panel.crop(fig)
        rgb = np.asarray(crop)
        saturation = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean()) / 255
        if saturation > 0.07:
            continue
        if len(find_bands(to_gray(crop))) < 3:
            continue
        result.append((panel, crop))
    return result


def analyze(data: bytes, filename: str = "upload", max_panels: int = 24) -> dict[str, Any]:
    t0 = time.time()
    is_pdf = filename.lower().endswith(".pdf") or data[:4] == b"%PDF"
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    provenance = ({"present": False, "status": "not_assessed",
                   "summary": "C2PA signatures for embedded PDF images were not checked."}
                  if is_pdf else read_c2pa(data, mime))
    figures = _figures_from_upload(data, filename)
    reuse = find_image_reuse([im for _, im in figures])
    report_figs, all_panels_gray, panel_refs = [], [], []
    n_panels = 0
    det = None
    for fi, (label, fig) in enumerate(figures):
        fig_entry = {
            "id": fi, "label": label, "width": fig.width, "height": fig.height,
            "image": _data_url(fig if max(fig.size) <= 1600 else fig.resize(
                (int(fig.width * 1600 / max(fig.size)), int(fig.height * 1600 / max(fig.size)))), "JPEG"),
            "scale": min(1.0, 1600 / max(fig.size)), "panels": [],
            "blot_screen": "not_assessed",
            "blot_note": "No confidently identified Western blot panels. Other image types are not scored by the Western blot CNN.",
        }
        for p, crop in _blot_candidates(fig):
            if n_panels >= max_panels:
                fig_entry["blot_note"] = "Panel analysis limit reached; remaining panels not assessed."
                break
            if det is None:
                det = detector()
            res = det.analyze(crop)
            all_panels_gray.append(to_gray(crop))
            panel_refs.append((fi, p.id))
            fig_entry["panels"].append({
                **p.to_dict(), "global_index": n_panels,
                "score": round(res.score, 3), "flagged": bool(res.score > det.threshold),
                "bands": res.bands, "heat": _data_url(_heat_overlay(res.heat, det.threshold)),
                "crop": _data_url(crop),
                "why": [{"feature": f, "value": round(v, 2)} for f, v in res.top_features],
            })
            fig_entry["blot_screen"] = "experimental"
            fig_entry["blot_note"] = "Experimental Western blot classifier; flags are not proof of manipulation."
            n_panels += 1
        report_figs.append(fig_entry)

    band_duplicates = find_duplicates(all_panels_gray) if all_panels_gray else []
    for d in band_duplicates:
        for side in ("a", "b"):
            fi, pid = panel_refs[d[side]["panel"]]
            d[side]["figure"], d[side]["panel_id"] = fi, pid
    flagged = [p for f in report_figs for p in f["panels"] if p["flagged"]]
    # Never call absence of findings "authentic" or "clean".
    verdict = "suspicious" if flagged or band_duplicates or reuse else ("no_blots" if n_panels == 0 else "no_findings")
    return {
        "filename": filename, "verdict": verdict,
        "summary": {
            "figures": len(report_figs), "panels": n_panels,
            "flagged_panels": len(flagged), "duplicate_pairs": len(band_duplicates),
            "image_reuse_pairs": len(reuse),
            "max_splice_score": round(max((p["score"] for f in report_figs for p in f["panels"]), default=0), 3),
            "threshold": round(det.threshold, 3) if det is not None else None,
            "content_credentials": provenance.get("present", False),
        },
        "provenance": provenance, "figures": report_figs,
        "image_reuse": reuse, "duplicates": band_duplicates,
        "checks": {
            "whole_image_reuse": "completed" if len(figures) >= 2 else "not_applicable_single_image",
            "local_region_reuse": "not_implemented",
            "blot_cnn": "experimental" if n_panels else "not_assessed",
            "c2pa": "not_assessed" if is_pdf else "checked_on_uploaded_file",
        },
        "synthid": {
            "status": "manual",
            "note": "SynthID testing is manual only. A negative result does not establish human origin. "
                    "Our small experiment found whole Gemini blots detectable but isolated bands too small to assess.",
            "url": "https://synthid.com",
        },
        "limitations": [
            "Identical and near-duplicate whole images may be legitimately reused; read the source captions.",
            "Whole-image matching does not detect copied patches within different figures.",
            "No universal fake-image or AI-origin model is being run.",
            "No finding is not evidence that an image is authentic.",
        ],
        "seconds": round(time.time() - t0, 2),
    }
