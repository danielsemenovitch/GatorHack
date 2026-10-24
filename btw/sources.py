"""Load the AI generated donor blots (cropped to the blot itself)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .imgio import load_rgb
from .panels import find_panels

AI_DIR = Path(__file__).resolve().parents[1] / "data" / "ai_sources"


def ai_panels(ai_dir: Path = AI_DIR) -> list[Image.Image]:
    out = []
    for f in sorted(ai_dir.glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        im = load_rgb(f)
        ps = find_panels(im, min_score=0.3)
        if ps:
            out.extend(p.crop(im) for p in ps if p.w >= 60 and p.h >= 20)
        else:
            out.append(im)
    return out
