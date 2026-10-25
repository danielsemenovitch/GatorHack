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

from .report import analyze

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
def analyze_upload(file: UploadFile = File(...), blot_mode: str = Form('auto'), cnn_version: str = Form('v3'), screening_policy: str = Form('benchmark')):
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
