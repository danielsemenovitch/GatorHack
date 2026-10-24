"""Generate AI western blots with the Gemini API to enlarge the AI donor pool.

Run on a laptop (needs internet and a key from https://aistudio.google.com):

    pip install google-genai pillow
    set GEMINI_API_KEY=...        (Windows)   |   export GEMINI_API_KEY=... (mac/linux)
    python laptop/gemini_blot_library.py --n 15

Images go to data/ai_sources/. Commit them, then rebuild the benchmark:
    python -m scripts.build_benchmark --skip-panels
    python -m scripts.train_detector

The model name is configurable because Google renames image models often.
Check the current list at https://ai.google.dev/gemini-api/docs/image-generation
"""
from __future__ import annotations

import argparse
import io
import os
import random
import time
from pathlib import Path

from PIL import Image

STYLES = [
    "exported from a digital chemiluminescence imager, smooth uniform light grey background",
    "scanned X-ray film, faint film grain, slightly uneven exposure",
    "fluorescent infrared imager (LI-COR style) converted to grayscale, dark bands on white",
    "high contrast exposure with some saturated bands",
    "faint bands with visible background speckles",
]
LAYOUTS = [
    "{n} lanes, one row of target protein bands and one row of loading control bands",
    "{n} lanes, a single horizontal strip cropped tightly around one band per lane",
    "{n} lanes, three stacked strips for three different antibodies",
]


def prompt(rng: random.Random) -> str:
    n = rng.choice([4, 5, 6, 8, 10, 12])
    return ("Generate an image: a western blot panel, " + rng.choice(LAYOUTS).format(n=n) + ", "
            + rng.choice(STYLES) + ". No labels, no text, no molecular weight marker, cropped tightly to the "
            "membrane like a panel in a published paper figure. Band intensities vary between lanes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "ai_sources")
    a = ap.parse_args()
    from google import genai  # pip install google-genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    made = 0
    while made < a.n:
        p = prompt(rng)
        try:
            resp = client.models.generate_content(model=a.model, contents=p)
        except Exception as e:
            print("request failed:", e); time.sleep(5); continue
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                im = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                f = a.out / f"gemini_api_{int(time.time())}_{made:02d}.png"
                im.save(f); made += 1
                print("saved", f.name, "|", p[:80])
                break
        time.sleep(2)


if __name__ == "__main__":
    main()
