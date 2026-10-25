"""Build examples/demo_reuse.pdf: a small PDF with one figure used twice.

Figure 1 and Figure 3 are the same CC BY figure on purpose. Figure 3 is resized
and recompressed, so the whole-image reuse check has to match it by appearance
rather than by identical bytes. Nothing here reflects anything the original
authors did; the title page says so.
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_A = ROOT / "data/real_figures/PMC10315040_Fig6.jpg"
FIG_B = ROOT / "data/real_figures/PMC10301924_pathogens1200801f002.jpg"
OUT = ROOT / "examples/demo_reuse.pdf"

COPY_SCALE = 0.95   # a different file at a different size; at 0.90 the glyph guard skips its two merged blot strips
COPY_QUALITY = 75

TITLE = (
    "Demo PDF assembled by the Below the Watermark team for Gator Hack V.\n\n"
    "Figure 1 and Figure 3 are the same published figure on purpose (Figure 3 is a resized, "
    "recompressed copy), so the image reuse check has something to find. Nothing here reflects "
    "anything the original authors did.\n\n"
    "Figures 1 and 3: Fig. 6 of PMC10315040 (doi:10.1186/s13048-023-01210-5), CC BY.\n"
    "Figure 2: Fig. 2 of PMC10301924 (doi:10.3390/pathogens12060801), CC BY."
)


def jpeg(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def add_figure(doc: fitz.Document, label: str, data: bytes, size: tuple[int, int]) -> None:
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 64), label, fontsize=14, fontname="helv")
    w, h = size
    box_w, box_h = 504, 640
    s = min(box_w / w, box_h / h)
    x0, y0 = 54, 86 + (box_h - h * s) / 2 if h * s < box_h * 0.9 else 86
    page.insert_image(fitz.Rect(x0, y0, x0 + w * s, y0 + h * s), stream=data)


def main() -> None:
    a = Image.open(FIG_A)
    b = Image.open(FIG_B)
    copy = a.resize((round(a.width * COPY_SCALE), round(a.height * COPY_SCALE)), Image.LANCZOS)

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(54, 72, 558, 400), TITLE, fontsize=11, fontname="helv")
    add_figure(doc, "Figure 1", jpeg(a, 92), a.size)
    add_figure(doc, "Figure 2", jpeg(b, 92), b.size)
    add_figure(doc, "Figure 3", jpeg(copy, COPY_QUALITY), copy.size)
    doc.set_metadata({"title": "Below the Watermark demo PDF", "author": "Below the Watermark team"})
    doc.save(OUT, garbage=4, deflate=True)
    print(f"wrote {OUT.relative_to(ROOT)}: figures {a.size}, {b.size}, copy {copy.size}")


if __name__ == "__main__":
    main()
