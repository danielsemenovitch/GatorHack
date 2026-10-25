"""FastAPI backend. Run:  uvicorn btw.api:app --port 8000"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import threading


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE lines from a git-ignored .env file (for GEMINI_API_KEY); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from . import region_match, search_agent
from .report import DEFAULT_MODEL, _data_url, _figures_from_upload, analyze

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WEB_DIST = ROOT / "web" / "dist"
EXAMPLES = ROOT / "examples"
MAX_BYTES = 25 * 1024 * 1024
log = logging.getLogger("btw.api")

app = FastAPI(title="Below the Watermark")
# The built frontend is served from this same origin, so browsers need no CORS
# for normal use. Only the Vite dev server (and origins listed in
# BTW_CORS_ORIGINS, comma separated) may call the API from another origin.
_ORIGINS = [o.strip() for o in os.environ.get(
    "BTW_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
def analyze_upload(file: UploadFile = File(...), blot_mode: str = Form('auto'), cnn_version: str = Form(DEFAULT_MODEL), screening_policy: str = Form('benchmark')):
    # A plain `def` endpoint runs in FastAPI's worker threadpool, so a long
    # CPU-bound analysis no longer blocks /api/health or other requests.
    data = file.file.read(MAX_BYTES + 1)  # never read more than the limit
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (25 MB max).")
    if not data:
        raise HTTPException(422, "Could not analyze this file: it is empty.")
    try:
        return JSONResponse(analyze(data, file.filename or "upload", blot_mode=blot_mode, cnn_version=cnn_version, screening_policy=screening_policy))
    except ValueError as e:  # our own validation messages are written for users
        raise HTTPException(422, f"Could not analyze this file: {e}")
    except Exception:  # do not leak internal exception text to the browser
        log.exception("analysis failed for %s", file.filename)
        raise HTTPException(422, "Could not analyze this file. It may be corrupt or in an unsupported format.")


# Build or load the reference library in the background, so the first upload
# does not wait for it.
threading.Thread(target=region_match.library, daemon=True).start()


@app.post("/api/search_web")
def search_web(file: UploadFile = File(...)):
    """The interpaper agent: search Europe PMC for papers that may contain these figures."""
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (25 MB max).")
    if not data:
        raise HTTPException(422, "The file is empty.")
    try:
        figures, _, _ = _figures_from_upload(data, file.filename or "upload")
        images, labels = [im for _, im in figures][:30], [lb for lb, _ in figures][:30]
        out = search_agent.run(data, file.filename or "upload", images, labels)
    except Exception:
        log.exception("web search failed for %s", file.filename)
        raise HTTPException(422, "The search could not run on this file.")
    for m in out["matches"]:
        x0, y0, x1, y1 = m["box"]
        m["crop"] = _data_url(images[m["figure"]].crop((x0, y0, x1, y1)).convert("RGB"), "JPEG")
        src = m.pop("source_image")
        sx0, sy0, sx1, sy1 = m["source_box"]
        m["source_crop"] = _data_url(src.crop((sx0, sy0, sx1, sy1)), "JPEG")
    return JSONResponse(out)


@app.get("/api/examples")
def examples():
    meta = EXAMPLES / "examples.json"
    return json.loads(meta.read_text()) if meta.exists() else []


@app.get("/api/examples/{name}")
def example_file(name: str):
    f = (EXAMPLES / name).resolve()
    if EXAMPLES.resolve() not in f.parents or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/results")
def results():
    out = {}
    for f in RESULTS.glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    return out


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
