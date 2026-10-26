"""Generate a varied set of AI Western blot images with the Gemini API, for research.

Purpose: AI donor images for the watermark-gap study (SynthID is applied by Google to
Gemini output) and for a generator-diverse benchmark. These are synthetic research
stimuli, labeled as such, never presented as real data.

Run from the repository root on a machine that can reach the Gemini API:
    python -m scripts.generate_ai_blots --n 60
The API key is read from GEMINI_API_KEY in the environment or the git-ignored .env file
and sent only in the x-goog-api-key header. Images are saved exactly as returned (no
re-encoding), so their SynthID watermark and any C2PA metadata are untouched.

Output: data/ai_donors_v6/gemini/<model>/<nnn>.<ext> and data/ai_donors_v6/gemini/manifest.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ai_donors_v6/gemini"
API = "https://generativelanguage.googleapis.com/v1beta"

IMAGING = [
    "scanned X-ray film from enhanced chemiluminescence, dark bands on a light grey background",
    "digital CCD chemiluminescence imager capture in grayscale, dark bands on a white background",
    "grayscale fluorescent Western blot scan with light bands on a dark background",
    "colorimetric Western blot on a nitrocellulose membrane, photographed, converted to grayscale",
]
LANES = [4, 6, 8, 10, 12]
PATTERN = [
    "one band per lane at the same height with clearly different intensities between lanes",
    "a band that becomes stronger from left to right across the lanes",
    "doublet bands in some lanes, single bands in others",
    "one strong band per lane plus a faint nonspecific band below it",
    "bands of equal intensity, as for a loading control",
]
QUALITY = [
    "clean background",
    "slightly uneven background with faint speckles",
    "mildly smiling bands and slight lane-to-lane width variation",
    "slightly overexposed bands with soft edges",
]
LAYOUT = [
    "a tight horizontal crop showing a single protein row",
    "two stacked rows: a target protein above and a loading control below",
    "three stacked rows of different proteins, each cropped tightly",
]


def load_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or "PASTE" in key.upper() or "your" in key.lower():
        raise SystemExit("GEMINI_API_KEY is not set (environment or .env)")
    return key


def call(method, path, key, body=None, timeout=120):
    req = urllib.request.Request(f"{API}/{path}", method=method,
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def image_models(key):
    names, token = [], None
    while True:
        res = call("GET", "models?pageSize=200" + (f"&pageToken={token}" if token else ""), key)
        for m in res.get("models", []):
            name = m["name"].split("/", 1)[1]
            if "image" in name and "generateContent" in m.get("supportedGenerationMethods", []) \
                    and "imagen" not in name and "embedding" not in name:
                names.append(name)
        token = res.get("nextPageToken")
        if not token:
            return names


def prompts(n, seed):
    combos = list(itertools.product(IMAGING, LANES, PATTERN, QUALITY, LAYOUT))
    random.Random(seed).shuffle(combos)
    for imaging, lanes, pattern, quality, layout in combos[:n]:
        yield {"imaging": imaging, "lanes": lanes, "pattern": pattern, "quality": quality, "layout": layout,
               "text": (f"A realistic Western blot image as it would appear in a biomedical figure panel: {layout}. "
                        f"{lanes} lanes, {pattern}. Imaging: {imaging}; {quality}. "
                        "Plain image only: no text, labels, molecular weight markers, numbers, borders or watermarks.")}


def generate(model, prompt, key):
    base = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "16:9"}}}
    for attempt in range(5):
        try:
            try:
                res = call("POST", f"models/{model}:generateContent", key, base)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                plain = {"contents": base["contents"], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
                res = call("POST", f"models/{model}:generateContent", key, plain)
            for cand in res.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    data = part.get("inlineData") or part.get("inline_data")
                    if data:
                        return base64.b64decode(data["data"]), data.get("mimeType") or data.get("mime_type"), None
            reason = (res.get("candidates") or [{}])[0].get("finishReason") or res.get("promptFeedback")
            return None, None, f"no image returned: {reason}"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            return None, None, f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 4:
                time.sleep(5)
                continue
            return None, None, f"network: {e}"
    return None, None, "gave up"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="images in total, split across the image models found")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", default="", help="comma-separated model ids; default: every Gemini image model available")
    a = ap.parse_args()
    key = load_key()
    models = [m for m in a.models.split(",") if m] or image_models(key)
    if not models:
        raise SystemExit("No Gemini image generation model is available to this key")
    print("Image models:", ", ".join(models), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {(r["model"], r["prompt_index"]) for r in manifest if r.get("file")}
    for i, p in enumerate(prompts(a.n, a.seed)):
        model = models[i % len(models)]
        if (model, i) in done:
            continue
        data, mime, err = generate(model, p["text"], key)
        rec = {"prompt_index": i, "model": model, "prompt": p, "utc": datetime.now(timezone.utc).isoformat(),
               "generator_family": "gemini", "watermark_expected": "SynthID (applied by Google)"}
        if data:
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "bin")
            f = OUT / model / f"{i:03d}.{ext}"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            rec.update(file=str(f.relative_to(ROOT)).replace("\\", "/"), mime=mime, bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest())
            print(f"[{i + 1}/{a.n}] {model} -> {f.name}", flush=True)
        else:
            rec["error"] = err
            print(f"[{i + 1}/{a.n}] {model} failed: {err}", flush=True)
        manifest = [r for r in manifest if (r["model"], r["prompt_index"]) != (model, i)] + [rec]
        manifest_path.write_text(json.dumps(manifest, indent=1))
        time.sleep(2)
    ok = sum(1 for r in manifest if r.get("file"))
    print(f"Done: {ok} images saved in {OUT.relative_to(ROOT)}; manifest at {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
