"""Do C2PA content credentials survive the steps a figure goes through on its way into a paper?

For every AI original (Gemini via the API, ChatGPT downloads), apply each step to the original
bytes and read C2PA with c2pa-python (btw.provenance.read_c2pa). Steps are ordinary operations
a researcher or a journal production system performs; none tries to remove metadata on purpose.

Writes results/c2pa_pipeline.json (per image and step) and prints survival counts.
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btw.provenance import read_c2pa  # noqa: E402


def mime_of(b: bytes) -> str:
    return "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ("image/jpeg" if b[:2] == b"\xff\xd8" else "image/tiff")


def pil_save(im, fmt, **kw):
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def steps(orig: bytes):
    im = Image.open(io.BytesIO(orig))
    im.load()
    fmt = "PNG" if mime_of(orig) == "image/png" else "JPEG"
    yield "original", orig
    yield "resave_same_format", pil_save(im, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    yield "jpeg_q95", pil_save(im.convert("RGB"), "JPEG", quality=95)
    yield "tiff", pil_save(im.convert("RGB"), "TIFF")
    w, h = im.size
    yield "crop_90pct", pil_save(im.crop((w // 20, h // 20, w - w // 20, h - h // 20)), fmt)
    canvas = Image.new("RGB", (w * 2 + 20, h * 2 + 20), "white")
    canvas.paste(im.convert("RGB"), (0, 0))
    yield "placed_in_2x2_figure", pil_save(canvas, "PNG")
    # PDF: embed the original bytes as a figure, then pull the embedded image back out,
    # and render the page (what a screenshot or a journal's page image would contain).
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(36, 36, 576, 36 + 540 * h / w), stream=orig)
    pdf = doc.tobytes()
    doc2 = fitz.open(stream=pdf, filetype="pdf")
    xref = doc2[0].get_images(full=True)[0][0]
    yield "pdf_embedded_then_extracted", doc2.extract_image(xref)["image"]
    yield "pdf_page_rendered", doc2[0].get_pixmap(dpi=150).tobytes("png")


def main():
    sets = {
        "gemini": [r for r in json.loads((ROOT / "data/ai_donors_v6/gemini/manifest.json").read_text()) if r.get("file")],
        "openai": json.loads(Path("/mnt/user-data/uploads/GatorHack/data/ai_donors_v6/openai/manifest.json").read_text()),
    }
    out, counts = [], defaultdict(lambda: [0, 0])
    for fam, recs in sets.items():
        for r in recs:
            p = ROOT / r["file"] if fam == "gemini" else Path("/mnt/user-data/uploads/GatorHack/data/ai_donors_v6/openai") / r["name"]
            orig = p.read_bytes()
            for step, b in steps(orig):
                c = read_c2pa(b, mime_of(b))
                ok = bool(c.get("present")) and bool(c.get("ai_generated"))
                out.append({"family": fam, "file": p.name, "model": r.get("model", "chatgpt"), "step": step,
                            "c2pa_present": bool(c.get("present")), "declares_ai": bool(c.get("ai_generated")), "bytes": len(b)})
                counts[(fam, step)][0] += ok
                counts[(fam, step)][1] += 1
    (ROOT / "results/c2pa_pipeline.json").write_text(json.dumps({"rows": out, "summary": {f"{k[0]}|{k[1]}": v for k, v in counts.items()}}, indent=1))
    for (fam, step), (k, n) in counts.items():
        print(f"{fam:7s} {step:28s} {k}/{n}")


if __name__ == "__main__":
    main()
