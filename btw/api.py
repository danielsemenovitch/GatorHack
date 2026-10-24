"""FastAPI backend. Run:  uvicorn btw.api:app --port 8000"""
from __future__ import annotations

import json
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

app = FastAPI(title="Below the Watermark")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), blot_mode: str = Form('auto'), cnn_version: str = Form('v3'), screening_policy: str = Form('conservative')):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (25 MB max).")
    try:
        return JSONResponse(analyze(data, file.filename or "upload", blot_mode=blot_mode, cnn_version=cnn_version, screening_policy=screening_policy))
    except Exception as e:  # surface a readable error to the UI
        raise HTTPException(422, f"Could not analyze this file: {e}")


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
