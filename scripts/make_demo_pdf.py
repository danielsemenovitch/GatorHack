"""Build examples/demo_reuse.pdf: a small "manuscript" with two subtle duplications.

Everything here is assembled by us from openly licensed (CC BY) figures, and the
title page says so. Nothing reflects anything the original authors did.

  Figure 1  Fig. 2 of PMC10755782, with ONE edit by us: the Vinculin strip in
            panel A (M10) is replaced by the HO 1 blot from our Figure 2,
            resized to fit. Intrapaper duplication: one blot shown as two proteins.
  Figure 2  Fig. 4 of PMC10352194, unaltered (the source of that strip).
  Figure 3  A cropped, resized and re-saved part of Fig. 1 of PMC10701888 (2023),
            presented as new data. Interpaper duplication: a published figure
            reused in a later paper. PMC10701888 is in the reference library.

The first page reads like a manuscript on the same topic as the source paper
(as paper mill products do), so the search agent has a realistic title and
captions to plan from; it does not name the source. The sources and our edits
are stated on the last page.
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data/real_figures"
OUT = ROOT / "examples/demo_reuse.pdf"

# Donor strip: HO 1 in panel b of PMC10352194 Fig. 4 (x, y, w, h in its pixels).
DONOR = (419, 313, 277, 55)
# Where it goes: over the M10 Vinculin strip in panel A of PMC10755782 Fig. 2.
TARGET_XY, TARGET_W = (54, 90), 168
# Part of PMC10701888 Fig. 1 (2023) reused as a new figure.
LIFT_BOX, LIFT_SCALE = (0, 150, 704, 332), 0.85

TITLE = "Fludarabine and clofarabine induce caspase 3 cleavage and a heat shock response in AML cells"
FRONT = (
    "DEMO MANUSCRIPT, NOT A REAL PAPER. Built by the Below the Watermark team for Gator Hack V from openly "
    "licensed figures, with two duplications we made on purpose. The last page lists the sources and our edits.\n\n"
    "Abstract. Combination chemotherapy with nucleoside analogs is used in acute myeloid leukemia (AML). We "
    "examined progranulin, heat shock proteins and markers of apoptosis (cleaved caspase 3, PARP1 cleavage and "
    "gamma-H2AX) by western blot in cells exposed to fludarabine and clofarabine."
)
CAPTIONS = [
    "Figure 1. Progranulin and vinculin over time and after treatment (western blot).",
    "Figure 2. Heat shock proteins and Nrf 2 / HO 1 after oxidative stress (western blot).",
    "Figure 3. Cleaved caspase 3, PARP1 and gamma-H2AX after combination treatment (western blot).",
]
BACK = (
    "Sources and our edits (all figures CC BY):\n"
    "- First figure: Fig. 2 of PMC10755782 (doi:10.1016/j.jbc.2023.105475). We replaced the M10 Vinculin strip "
    "in panel A with the HO 1 blot from the second figure, resized. This is the within-paper duplication.\n"
    "- Second figure: Fig. 4 of PMC10352194 (doi:10.1007/s12192-023-01349-6), unaltered.\n"
    "- Third figure: part of Fig. 1 of PMC10701888 (doi:10.3389/fonc.2023.1287444), cropped and resized by us. "
    "This is the across-paper duplication.\n\n"
    "Nothing here reflects anything the original authors did; their published figures are unaltered."
)


def jpeg(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def add_figure(doc: fitz.Document, caption: str, data: bytes, size: tuple[int, int]) -> None:
    page = doc.new_page(width=612, height=792)
    w, h = size
    s = min(504 / w, 600 / h)
    page.insert_image(fitz.Rect(54, 72, 54 + w * s, 72 + h * s), stream=data)
    page.insert_textbox(fitz.Rect(54, 72 + h * s + 12, 558, 760), caption, fontsize=10, fontname="helv")


def main() -> None:
    f1 = Image.open(FIG / "PMC10755782_fig2.jpg").convert("RGB")
    f2 = Image.open(FIG / "PMC10352194_Fig4.jpg").convert("RGB")
    lib = Image.open(FIG / "PMC10701888_f1.jpg").convert("RGB")

    x, y, w, h = DONOR
    strip = f2.crop((x, y, x + w, y + h))
    strip = strip.resize((TARGET_W, round(h * TARGET_W / w)), Image.LANCZOS)
    fig1 = f1.copy()
    fig1.paste(strip, TARGET_XY)

    lifted = lib.crop(LIFT_BOX)
    lifted = lifted.resize((round(lifted.width * LIFT_SCALE), round(lifted.height * LIFT_SCALE)), Image.LANCZOS)

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(54, 72, 558, 140), TITLE, fontsize=16, fontname="hebo")
    page.insert_textbox(fitz.Rect(54, 150, 558, 500), FRONT, fontsize=11, fontname="helv")
    add_figure(doc, CAPTIONS[0], jpeg(fig1, 92), fig1.size)
    add_figure(doc, CAPTIONS[1], jpeg(f2, 92), f2.size)
    add_figure(doc, CAPTIONS[2], jpeg(lifted, 85), lifted.size)
    back = doc.new_page(width=612, height=792)
    back.insert_textbox(fitz.Rect(54, 72, 558, 700), BACK, fontsize=11, fontname="helv")
    doc.set_metadata({"title": TITLE, "author": ""})
    doc.save(OUT, garbage=4, deflate=True)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
