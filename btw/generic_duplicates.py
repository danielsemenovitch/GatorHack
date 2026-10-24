"""Conservative, document-local image reuse checks for arbitrary scientific figures.

The checks establish visual reuse, not fabrication or research misconduct.
Whole-image only: cropped/moved internal patches are NOT covered here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass
class Fingerprint:
    pixels: str
    phash: int
    gray: np.ndarray
    information: float


def fingerprint(image: Image.Image) -> Fingerprint:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    # A hash of decoded pixels finds identical images across containers/metadata.
    exact = hashlib.sha256(f"{width}x{height}|".encode() + rgb.tobytes()).hexdigest()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].flatten()[1:]  # omit DC component
    median = float(np.median(block))
    bits = 0
    for bit in block > median:
        bits = (bits << 1) | int(bit)
    preview = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= max(1, gray.size)
    p = hist[hist > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return Fingerprint(exact, bits, preview, entropy)


def find_image_reuse(images: list[Image.Image], max_comparisons: int = 4000) -> list[dict]:
    """Return verified whole-image exact/near duplicates, with source figure indices.

    Pair candidates are selected by DCT perceptual hash, then confirmed using
    resized grayscale structural similarity. Deliberately ignores crops,
    rotations, inside-figure patches, and cross-document/global plagiarism.
    """
    keys = [fingerprint(img) for img in images]
    findings = []
    comparisons = 0
    for i, a in enumerate(keys):
        for j in range(i + 1, len(keys)):
            if comparisons >= max_comparisons:
                return findings
            comparisons += 1
            b = keys[j]
            base = {
                "a": {"figure": i}, "b": {"figure": j},
                "review_required": True,
                "scope": "whole embedded image; same uploaded file",
                "limitations": "Reused images may be legitimately repeated; compare captions and methods. "
                               "This check does not examine copied regions inside different images.",
            }
            if a.pixels == b.pixels:
                findings.append({
                    **base, "method": "exact_pixel_duplicate", "similarity": 1.0,
                    "note": "Identical decoded RGB pixels; file metadata may differ.",
                    "low_information": a.information < 2.0,
                })
                continue
            # Uniform figure backgrounds and tiny-detail templates trigger false matches.
            if min(a.information, b.information) < 2.0:
                continue
            distance = (a.phash ^ b.phash).bit_count()
            if distance > 8:
                continue
            score = float(structural_similarity(a.gray, b.gray, data_range=255))
            # Hash screening alone is insufficient; require strong full-frame agreement.
            if score >= 0.92:
                findings.append({
                    **base, "method": "near_duplicate", "similarity": round(score, 4),
                    "phash_distance": distance,
                    "note": "Similar image composition after alignment/resize; possible resave or rescaling.",
                    "low_information": False,
                })
    return findings
