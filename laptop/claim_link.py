"""Link a flagged figure to the claims in the paper that rely on it.

Run on a laptop with a Gemini API key:

    pip install google-genai pymupdf
    python laptop/claim_link.py paper.pdf --figure 3

Prints JSON: the figure caption and every sentence in the paper that cites
the figure, with a one line statement of what each claim depends on. Uses
Gemini only to read; it never judges whether the paper is fraudulent.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pymupdf

PROMPT = """You are helping a reviewer. Below is the full text of a paper.
Figure {fig} contains a western blot panel that an image forensics tool flagged
for a closer look (the flag is NOT proof of misconduct).

Return JSON with keys:
  "caption": the caption of Figure {fig} (verbatim, trimmed),
  "claims": a list of objects {{"quote": exact sentence from the text that cites Figure {fig},
            "section": section name, "depends_on": one short sentence saying which result in the
            figure this claim relies on}}.
Only include sentences that explicitly cite Figure {fig} (e.g. "Fig. {fig}", "Figure {fig}B").
Return JSON only.

PAPER TEXT:
{text}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--figure", required=True)
    ap.add_argument("--model", default=os.environ.get("GEMINI_TEXT_MODEL", "gemini-3-pro-preview"))
    a = ap.parse_args()
    doc = pymupdf.open(a.pdf)
    text = "\n".join(p.get_text() for p in doc)[:400_000]
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(model=a.model, contents=PROMPT.format(fig=a.figure, text=text))
    raw = resp.text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    try:
        print(json.dumps(json.loads(raw), indent=1))
    except json.JSONDecodeError:
        print(raw)


if __name__ == "__main__":
    main()
