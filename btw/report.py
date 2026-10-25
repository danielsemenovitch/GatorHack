"""Inspect raster figures / PDFs with general reuse checks and optional blot screening.

This tool generates leads for HUMAN review, not authenticity or misconduct verdicts.
"""
from __future__ import annotations

import logging
from pathlib import Path
import os
import json
import re

import base64
import io
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .bands import find_bands
from .duplicates import find_duplicates
from .generic_duplicates import find_image_reuse
from . import region_match

log = logging.getLogger("btw.report")
from .imgio import load_rgb, to_gray
from .panels import find_panels
from .provenance import read_c2pa

_DETECTORS = {}


def detector(version: str = "v3"):
    if version not in _DETECTORS:
        from .detector import load_best
        _DETECTORS[version] = load_best(version)
    return _DETECTORS[version]


def _data_url(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _heat_overlay(heat: np.ndarray, threshold: float) -> Image.Image:
    # Apply review cutoff BEFORE softening the overlay. Previously, a small
    # hot patch blurred below threshold and disappeared altogether.
    raw = np.clip((np.clip(heat, 0, 1).astype(np.float32) - threshold) /
                  max(1e-3, 1 - threshold), 0, 1) ** 0.7
    a = cv2.GaussianBlur(raw, (0, 0), 1.5)
    a = np.clip(a, 0, 1)
    rgba = np.zeros((*a.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 200, 40, 95
    rgba[..., 3] = (a * 190).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


MAX_PIXELS = 12_000_000
MAX_TOTAL_PIXELS = 150_000_000  # per request, so one PDF cannot exhaust server memory

# CNN v3 was trained on Europe PMC web figures: median width 751 px
# (10th to 90th percentile 682 to 791 px). Publisher PDFs often embed the same
# figures at 1,000 to 2,000+ px, which the model never saw. On held-out CC BY
# test panels, enlarging clean panels 2.5x tripled clean false flags at the
# benchmark threshold (audit, Oct 24). Blot screening therefore runs on a copy
# of each figure resized to the training width; display uses the original.
ANALYSIS_WIDTH = 750
ANALYSIS_TRIGGER = 1.25  # only resize figures wider than 1.25x the training width


def _pixmap_rgb(pix) -> Image.Image:
    """Convert any PyMuPDF pixmap (gray, CMYK, indexed, alpha) to RGB."""
    import pymupdf as fitz
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)  # drop alpha
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _figures_from_upload(data: bytes, filename: str, max_figures: int = 48):
    """Return (figures, skipped, repeats).

    figures: [(label, RGB image)] in page order.
    skipped: images that were NOT analyzed, with a reason (never silent).
    repeats: label -> later placements of a page decoration, i.e. one embedded
             PDF object drawn at the same spot on 3+ pages (journal logo,
             header). Those are reported once instead of producing a
             pairwise "reuse" finding on every page. Any other repeated
             placement of the same object stays a separate figure, so
             genuine reuse of one image in two figures is still flagged.
    """
    skipped: list[dict] = []
    repeats: dict[str, list[str]] = {}
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf as fitz
        doc = fitz.open(stream=data, filetype="pdf")
        placements = []  # (label, xref, w, h, page_no, rect)
        for pno, page in enumerate(doc):
            for k, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    rects = page.get_image_rects(xref)
                    rect = tuple(round(v) for v in rects[0]) if rects else None
                except Exception:
                    rect = None
                placements.append((f"page {pno + 1}, image {k + 1}", xref, int(img[2]), int(img[3]), pno, rect))
        by_xref: dict[int, list] = {}
        for pl in placements:
            by_xref.setdefault(pl[1], []).append(pl)
        decoration = {x for x, pls in by_xref.items()
                      if len({pl[4] for pl in pls}) >= 3 and len({pl[5] for pl in pls}) == 1 and pls[0][5] is not None}
        out, decoded, first = [], {}, {}
        budget = MAX_TOTAL_PIXELS
        for label, xref, w, h, pno, rect in placements:
            if xref in decoration and xref in first:
                repeats.setdefault(first[xref], []).append(label)
                continue
            first.setdefault(xref, label)
            if w < 150 or h < 80:
                skipped.append({"label": label, "reason": "too_small", "width": w, "height": h})
                continue
            if w * h > MAX_PIXELS:  # checked BEFORE decoding (decompression bombs)
                skipped.append({"label": label, "reason": "too_large", "width": w, "height": h})
                continue
            if len(out) >= max_figures:
                skipped.append({"label": label, "reason": "figure_limit", "width": w, "height": h})
                continue
            if xref not in decoded and w * h > budget:
                skipped.append({"label": label, "reason": "pixel_budget", "width": w, "height": h})
                continue
            if xref not in decoded:
                budget -= w * h
                try:
                    decoded[xref] = (_pixmap_rgb(fitz.Pixmap(doc, xref)), "")
                except Exception as e:  # one odd image must not sink the whole PDF
                    decoded[xref] = (None, str(e)[:120])
            im, err = decoded[xref]
            if im is None:
                skipped.append({"label": label, "reason": "unreadable", "detail": err})
                continue
            out.append((label, im))
        doc.close()
        return out, skipped, repeats
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        raise ValueError("This file is not a readable PNG, JPEG or PDF.") from None
    if im.width * im.height > MAX_PIXELS:  # header only; nothing decoded yet
        raise ValueError("Image exceeds the 12 megapixel analysis limit.")
    try:
        im = load_rgb(im)
    except Exception:
        raise ValueError("This image could not be decoded (it may be truncated or corrupt).") from None
    return [("uploaded image", im)], skipped, repeats


def _analysis_scale(fig: Image.Image) -> float:
    """Factor that brings a figure to the CNN's training width (1.0 if already close)."""
    if fig.width <= ANALYSIS_WIDTH * ANALYSIS_TRIGGER:
        return 1.0
    return ANALYSIS_WIDTH / fig.width


def _score_at_training_scale(det, crop: Image.Image, scale: float):
    """Run the CNN on `crop` resized by `scale`; return results in crop pixels."""
    if scale >= 1.0:
        return det.analyze(crop)
    small = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    res = det.analyze(small)
    heat = cv2.resize(np.asarray(res.heat, np.float32), (crop.width, crop.height), interpolation=cv2.INTER_LINEAR)
    sx, sy = crop.width / small.width, crop.height / small.height
    bands = [{**b, "x": int(round(b["x"] * sx)), "y": int(round(b["y"] * sy)),
              "w": max(1, int(round(b["w"] * sx))), "h": max(1, int(round(b["h"] * sy)))} for b in res.bands]
    return SimpleNamespace(score=res.score, heat=heat, bands=bands, top_features=res.top_features)


def _horizontal_blot_structure(crop: Image.Image, min_ratio: float = 1.55) -> bool:
    """Reject isotropic cellular textures that happen to resemble dark bands.

    Western blot lanes have predominantly horizontal, elongated edges.
    Compare vertical-vs-horizontal gradients AFTER blurring so cell boundaries
    do not gain an advantage from high-frequency microscopy detail.
    This is a routing guard, not a validated modality classifier.
    """
    gray = to_gray(crop).astype(np.uint8)
    if min(gray.shape) < 16:
        return False
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    horizontal_edge_ratio = float(np.mean(np.abs(gy))) / (float(np.mean(np.abs(gx))) + 1e-3)
    return horizontal_edge_ratio >= min_ratio


def _text_heavy(crop: Image.Image, scale: float = 1.0) -> bool:
    """Reject framed tables and annotation boxes (many small dark glyphs).

    Same glyph count used by scripts/build_benchmark.filter_panels to keep the
    training panels (text heavy panels were excluded there), measured at the
    training scale so high resolution figures are judged like web figures.
    """
    if scale < 1.0:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    g = to_gray(crop)
    dark = (g < 90).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    glyphs = sum(1 for i in range(1, n) if 4 <= st[i, 4] <= 250 and st[i, 2] <= 25 and st[i, 3] <= 25)
    density = glyphs / max(1.0, g.size / 10000)
    white = float((g > 235).mean())
    return density >= 25 or (white > 0.45 and density > 6)


# Flag lines on the display index (0.50 = the calibrated Standard line).
# Strict: the lowest line on a fixed grid that flags at most 1% of clean panels
# on calibration papers (scripts/calibrate_strict.py, results/strict_line.json).
# On 20 held-out test papers it caught 24% of pasted AI bands and flagged 2.5%
# of clean panels; Standard caught 33% and flagged 5.5%. The old 0.90 line caught 4%.
STRICT_LINE = 0.60
CHANCE_FLAG_RATE = {"benchmark": 0.05, "conservative": 0.025}

# The model the site uses. A newer model (models/splice_cnn_v4.npz, same format
# as v3) is switched on with BTW_MODEL=v4 once its strict line is calibrated:
#   python -m scripts.calibrate_strict --model v4   -> results/strict_line_v4.json
DEFAULT_MODEL = os.environ.get("BTW_MODEL", "v3")


def strict_line(model: str) -> tuple[float, float]:
    """(strict line, chance flag rate at it) for a model; v3's are fixed above."""
    if model == "v3":
        return STRICT_LINE, CHANCE_FLAG_RATE["conservative"]
    f = Path(__file__).resolve().parents[1] / "results" / f"strict_line_{model}.json"
    if f.exists():
        rec = json.loads(f.read_text())
        line = float(rec["chosen_line"])
        return line, float(rec["test"][str(rec["chosen_line"])]["clean_flagged"])
    return STRICT_LINE, CHANCE_FLAG_RATE["conservative"]


def _blot_candidates(fig: Image.Image, mode: str = "auto"):
    """Gate the experimental blot model; manual mode is user-confirmed, NOT validated.

    'auto': conservative blot likeness. 'blot': user explicitly identifies blot,
    so less restrictive technical criteria apply. 'off': never call blot CNN.
    """
    if mode == "off":
        return []
    user_selected = mode == "blot"
    result = []
    for panel in find_panels(fig, min_score=0.55 if user_selected else 0.75):
        if panel.blot_score < (0.64 if user_selected else 0.82):
            continue
        if panel.w < (60 if user_selected else 90) or panel.h < 20:
            continue
        if panel.w / max(1, panel.h) < (1.2 if user_selected else 1.5):
            continue
        crop = panel.crop(fig)
        rgb = np.asarray(crop)
        saturation = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean()) / 255
        if saturation > (0.12 if user_selected else 0.07):
            continue
        if len(find_bands(to_gray(crop))) < (2 if user_selected else 3):
            continue
        if not _horizontal_blot_structure(crop, min_ratio=1.20 if user_selected else 1.55):
            continue
        if _text_heavy(crop, _analysis_scale(fig)):
            continue
        result.append((panel, crop))
    return result


def analyze(data: bytes, filename: str = "upload", max_panels: int = 64,
            blot_mode: str = "auto", cnn_version: str | None = None,
            screening_policy: str = "benchmark") -> dict[str, Any]:
    if blot_mode not in ("auto", "blot", "off"):
        raise ValueError("blot_mode must be 'auto', 'blot', or 'off'")
    cnn_version = cnn_version or DEFAULT_MODEL
    if not re.fullmatch(r"v\d+", cnn_version):
        raise ValueError("cnn_version must look like v3")
    if screening_policy not in ("conservative", "benchmark"):
        raise ValueError("screening_policy must be conservative or benchmark")
    if not (1 <= max_panels <= 256):
        raise ValueError("max_panels must be between 1 and 256")
    t0 = time.time()
    is_pdf = filename.lower().endswith(".pdf") or data[:4] == b"%PDF"
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    provenance = ({"present": False, "status": "not_assessed",
                   "summary": "C2PA signatures for embedded PDF images were not checked."}
                  if is_pdf else read_c2pa(data, mime))
    figures, skipped_images, repeats = _figures_from_upload(data, filename)
    reuse = find_image_reuse([im for _, im in figures])
    # Shared regions: between figures of this upload (intrapaper) and against the
    # reference library of published figures (interpaper). Both use the same
    # verified keypoint matcher, so a cropped, resized or flipped copy still matches.
    fig_images = [im for _, im in figures]
    match_feats = [region_match.features(im, with_flip=True) for im in fig_images[:30]]
    whole_pairs = {frozenset((m["a"]["figure"], m["b"]["figure"])) for m in reuse}
    region_reuse = [m for m in (region_match.within_upload(fig_images, match_feats) if len(figures) > 1 else [])
                    if frozenset((m["a"]["figure"], m["b"]["figure"])) not in whole_pairs]
    try:
        library_matches = region_match.against_library(fig_images[:30], match_feats)
        library_info = region_match.library_size()
    except Exception:  # a missing or unreadable library must never break an analysis
        log.exception("reference library check failed")
        library_matches, library_info = [], None
    for m in region_reuse:
        for side in ("a", "b"):
            x0, y0, x1, y1 = m[side]["box"]
            m[side]["label"] = figures[m[side]["figure"]][0]
            m[side]["crop"] = _data_url(fig_images[m[side]["figure"]].crop((x0, y0, x1, y1)).convert("RGB"), "JPEG")
    for m in library_matches:
        x0, y0, x1, y1 = m["box"]
        m["label"] = figures[m["figure"]][0]
        m["crop"] = _data_url(fig_images[m["figure"]].crop((x0, y0, x1, y1)).convert("RGB"), "JPEG")
        try:
            src = Image.open(region_match.LIBRARY_DIR / m["source"]["file"]).convert("RGB")
            sx0, sy0, sx1, sy1 = m["source_box"]
            m["source_crop"] = _data_url(src.crop((sx0, sy0, sx1, sy1)), "JPEG")
        except Exception:
            m["source_crop"] = None
        pm = m["source"].get("pmcid") or ""
        m["source"]["url"] = f"https://europepmc.org/article/PMC/{pm.removeprefix('PMC')}" if pm else None
    report_figs, all_panels_gray, panel_refs = [], [], []
    n_panels = 0
    skipped_panels = 0
    det = None
    effective_threshold = None
    for fi, (label, fig) in enumerate(figures):
        # Panels are FOUND on the original figure (panel routing was tuned
        # there and keeps adjacent strips apart). Each panel crop is then
        # SCORED by the CNN on a copy resized to the training resolution;
        # heat and band boxes are mapped back to original panel pixels, so
        # every coordinate in the report is in original figure pixels.
        analysis_scale = _analysis_scale(fig)
        fig_entry = {
            "id": fi, "label": label, "width": fig.width, "height": fig.height,
            "native_width": fig.width, "native_height": fig.height,
            "analysis_scale": round(analysis_scale, 4),
            "image": _data_url(fig if max(fig.size) <= 1600 else fig.resize(
                (int(fig.width * 1600 / max(fig.size)), int(fig.height * 1600 / max(fig.size)))), "JPEG"),
            "scale": min(1.0, 1600 / max(fig.size)), "panels": [],
            "repeated_placements": repeats.get(label, []),
            "blot_screen": "not_assessed",
            "blot_note": "No confidently identified Western blot panels. Other image types are not scored by the Western blot CNN.",
        }
        candidates = _blot_candidates(fig, blot_mode)
        for candidate_i, (p, crop) in enumerate(candidates):
            if n_panels >= max_panels:
                skipped_panels += len(candidates) - candidate_i
                fig_entry["blot_note"] = "Panel analysis limit reached; remaining candidate panels not assessed."
                fig_entry["blot_screen"] = "partial" if fig_entry["panels"] else "not_assessed_limit"
                break
            if det is None:
                det = detector(cnn_version)
                # Standard is the calibrated research line; Strict is STRICT_LINE
                # (see above). Neither is a probability of misconduct. V2
                # saturates near 1 and has no defensible strict cutoff, so it
                # keeps its original threshold.
                effective_threshold = (strict_line(cnn_version)[0] if (cnn_version != "v2" and
                                       screening_policy == "conservative")
                                       else float(det.threshold))
            res = _score_at_training_scale(det, crop, analysis_scale)
            all_panels_gray.append(to_gray(crop))
            panel_refs.append((fi, p.id))
            fig_entry["panels"].append({
                **p.to_dict(), "global_index": n_panels,
                "score": round(float(res.score), 6), "flagged": bool(res.score > effective_threshold),
                "bands": res.bands, "heat": _data_url(_heat_overlay(res.heat, effective_threshold)),
                "heat_basis": "overlapping 32px CNN patch anomaly estimates, not segmentation or edit boundaries",
                "crop": _data_url(crop),
                "why": [{"feature": f, "value": round(v, 2)} for f, v in res.top_features],
            })
            fig_entry["blot_screen"] = "experimental"
            fig_entry["blot_note"] = (
                "User-selected Western blot screening; experimental results must be independently checked."
                if blot_mode == "blot" else
                "Experimental Western blot classifier; flags are not proof of manipulation."
            )
            n_panels += 1
        report_figs.append(fig_entry)

    # Higher-precision *review candidate* operating point. The original
    # library's 0.5 noise threshold produced many benign matches on genuine
    # published blot strips. This intentionally sacrifices recall to avoid
    # overwhelming reviewers with weak matches.
    band_duplicates = (find_duplicates(all_panels_gray, threshold=0.90, shape_floor=0.90)
                       if all_panels_gray else [])
    for d in band_duplicates:
        # d.b.id == -1 is a matched image location, NOT a detected band.
        for side in ("a", "b"):
            loc = d[side]
            fi, pid = panel_refs[loc["panel"]]
            loc["figure"], loc["panel_id"] = fi, pid
            loc["figure_label"] = report_figs[fi]["label"]
            gray = all_panels_gray[loc["panel"]]
            hh, ww = gray.shape
            x0 = max(0, int(loc["x"]) - 4)
            y0 = max(0, int(loc["y"]) - 4)
            x1 = min(ww, int(loc["x"] + loc["w"]) + 4)
            y1 = min(hh, int(loc["y"] + loc["h"]) + 4)
            loc["crop"] = _data_url(Image.fromarray(gray[y0:y1, x0:x1].astype(np.uint8)))
            loc["is_detected_band"] = bool(isinstance(loc.get("id"), int) and loc["id"] >= 0)
    # Band matches between two figures that are already a whole-image reuse pair
    # restate that finding (every band of a copied figure matches its twin), so
    # they are counted under the reuse pair instead of listed again.
    reuse_pairs = {frozenset((m["a"]["figure"], m["b"]["figure"])) for m in reuse}
    folded = [d for d in band_duplicates if frozenset((d["a"]["figure"], d["b"]["figure"])) in reuse_pairs
              and d["a"]["figure"] != d["b"]["figure"]]
    band_duplicates = [d for d in band_duplicates if d not in folded]
    for m in reuse:
        m["matching_bands"] = sum(1 for d in folded if frozenset((d["a"]["figure"], d["b"]["figure"])) == frozenset((m["a"]["figure"], m["b"]["figure"])))
    flagged =[p for f in report_figs for p in f["panels"] if p["flagged"]]
    ai_declared = bool(provenance.get("present") and provenance.get("ai_generated"))
    # Never call absence of findings "authentic" or "clean".
    verdict = ("suspicious" if flagged or band_duplicates or reuse or region_reuse or library_matches or ai_declared
               else ("no_blots" if n_panels == 0 else "no_findings"))
    per_figure = [
        {"figure": f["id"], "label": f["label"],
         "analyzed_panels": len(f["panels"]),
         "flagged_panels": sum(bool(p["flagged"]) for p in f["panels"])}
        for f in report_figs
    ]
    # Many flags in one figure may share an ordinary imaging/processing cause.
    widespread = [r for r in per_figure if r["analyzed_panels"] >= 4
                  and r["flagged_panels"] >= 3
                  and r["flagged_panels"] / r["analyzed_panels"] >= 0.5]
    return {
        "filename": filename, "verdict": verdict, "blot_mode": blot_mode, "cnn_version": cnn_version,
        "screening_policy": screening_policy,
        "summary": {
            "figures": len(report_figs), "panels": n_panels,
            "skipped_panels": skipped_panels, "max_panels": max_panels,
            "per_figure": per_figure, "widespread_flag_figures": widespread,
            "flagged_panels": len(flagged), "duplicate_pairs": len(band_duplicates),
            "image_reuse_pairs": len(reuse),
            "region_reuse_pairs": len(region_reuse),
            "library_matches": len(library_matches),
            "library": library_info,
            "max_splice_score": round(max((p["score"] for f in report_figs for p in f["panels"]), default=0), 6),
            "threshold": round(float(effective_threshold), 6) if det is not None else None,
            "benchmark_threshold": round(float(det.threshold), 6) if det is not None else None,
            "screening_policy": screening_policy,
            "chance_flag_rate": ((strict_line(cnn_version)[1] if screening_policy == "conservative" else CHANCE_FLAG_RATE["benchmark"])
                                 if cnn_version != "v2" else None),
            "policy_note": (f"Strict line {strict_line(cnn_version)[0]:.2f}: set on calibration papers to flag at most 1% of clean panels."
                            if cnn_version != "v2" and screening_policy == "conservative" else
                            "V2 output saturates at 1.0; stricter cutoffs are not defensibly calibrated, so v2 retains its legacy benchmark threshold."
                            if cnn_version == "v2" and screening_policy == "conservative" else
                            "Standard line 0.50 (default): set so about 5% of clean panels from separate calibration papers are flagged. Measured on simulated edits, not validated on whole real papers."),
            "score_kind": ("threshold-centered display index" if getattr(det, "space", None) == "logit"
                           else "uncalibrated CNN patch score") if det is not None else None,
            "score_is_probability": False,
            "model_version": cnn_version if n_panels else None,
            "content_credentials": provenance.get("present", False),
            "credential_declares_ai": ai_declared,
            "skipped_images": len(skipped_images),
            "repeated_object_placements": sum(len(v) for v in repeats.values()),
            "resized_for_analysis": sum(1 for f in report_figs if f["analysis_scale"] < 1 and f["panels"]),
            "analysis_width": ANALYSIS_WIDTH,
        },
        "provenance": provenance, "figures": report_figs,
        "image_reuse": reuse, "duplicates": band_duplicates,
        "region_reuse": region_reuse, "library_matches": library_matches,
        "skipped_images": skipped_images[:100],
        "checks": {
            "whole_image_reuse": "completed" if len(figures) >= 2 else "not_applicable_single_image",
            "local_region_reuse": "not_implemented",
            "blot_cnn": ("partially_assessed" if skipped_panels else "experimental") if n_panels
                        else ("not_assessed_limit" if skipped_panels else "not_assessed"),
            "blot_panel_limit": "reached" if skipped_panels else "not_reached",
            "blot_mode": blot_mode,
            "cnn_version": cnn_version,
            "screening_policy": screening_policy,
            "heat_localization": "patch_level_32px_not_pixel_boundaries",
            "band_duplicate_policy": "conservative_noise_0.90_shape_0.90",
            "c2pa": "not_assessed" if is_pdf else "checked_on_uploaded_file",
            "blot_resolution": f"panels from figures wider than {int(ANALYSIS_WIDTH * ANALYSIS_TRIGGER)} px are scored on a copy resized to a {ANALYSIS_WIDTH} px wide figure (training resolution)",
            "skipped_images": "none" if not skipped_images else f"{len(skipped_images)} not analyzed (see skipped_images)",
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
            "The user-confirmed Western blot mode relaxes a heuristic gate; it does not validate modality classification.",
            "No universal fake-image or AI-origin model is being run.",
            "The CNN score is an uncalibrated anomaly index, not a percentage probability of manipulation.",
            "Conservative screening uses a stricter unvalidated operating point and will miss more genuine edits.",
            "Blot heat shows overlapping 32-pixel model patches, not localized proof or pixel-accurate boundaries.",
            "Multiple flags in one figure may share a benign imaging or processing cause.",
            "A copied-band candidate's matched location is not necessarily a separately detected band.",
            "No finding is not evidence that an image is authentic.",
            f"The blot CNN was trained on ~{ANALYSIS_WIDTH} px wide web figures; larger figures are screened on a resized copy, so edits made at full resolution may be harder to detect.",
            "The same embedded PDF object drawn on several pages (for example a journal logo) is reported once, not as image reuse.",
        ],
        "seconds": round(time.time() - t0, 2),
    }
