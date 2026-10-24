# Devpost submission draft

**Title:** Below the Watermark

**Tagline:** AI watermarks catch whole fake images. Science gets faked one band at a time.

## Inspiration

I work in a wet lab, and western blots are the evidence behind a huge share of biomedical claims. In the past year Gemini's image tools made it trivial to fabricate a blot, or to make one band darker or emptier, and Google answered with SynthID: an invisible watermark plus a public detector opened on October 7, 2026. We wanted to know whether that safety net works for science.

## What it does

- **The finding.** We tested SynthID Detector by hand on AI made blots. Whole AI images were caught every time, even after JPEG, resizing, cropping and PDF export. The visible content credential died on the first re-save. Single AI bands were "not enough details." [Pasted band result from results/synthid_manual.json rows S1 to S4, stated exactly.]
- **The benchmark.** [N, from the README] labeled blot images from [P] open access (CC BY) papers, forged five ways (including AI splices from Gemini), each run through realistic publishing steps, each with a pixel mask showing exactly what was changed.
- **The tool.** Upload a figure or a paper PDF. It finds every blot panel and band, scores each band against the rest of its own blot, looks for copied bands, reads C2PA credentials, and hands you the panel to check in SynthID.

## How we built it

- Python: OpenCV, scikit-image and PyMuPDF for panels, bands and publishing simulation.
- A gradient boosted baseline on hand built forensic features, plus a small CNN written in JAX and trained on CPU, both judging each region relative to its own panel.
- `c2pa-python` for content credentials.
- FastAPI backend and a React frontend designed like a film lightbox.
- Europe PMC's API for openly licensed figures.
- Gemini for the AI blots, SynthID Detector for the watermark tests (by hand, within its terms).

## Challenges

- **Resolution.** Europe PMC and PubMed Central serve figures about 750 pixels wide, so a band can be 13 × 6 pixels. Our first detector learned "this is a band" instead of "this band was edited." We fixed that with hard negatives and by comparing every band with its neighbors, and we report the honest numbers.
- **Terms of service.** SynthID Detector forbids automated or adversarial use, so every test was a fixed scenario uploaded by hand, and our adversarial training targets our own detector, not Google's.
- **Data rights.** Every real figure is openly licensed (CC BY) and attributed, every edit is ours, and no unpublished lab data is used anywhere.

## Accomplishments

- A measured, reproducible gap in a two week old public AI safety tool.
- An open benchmark with pixel level ground truth for AI era blot fraud.
- An end to end tool, from PDF to flagged band, with plain language explanations.

## What we learned

Watermarks are strong where they are applied (whole images) and absent where science is vulnerable (small regions). Provenance has to be layered.

## What's next

- Full resolution figures from publishers, which carry far more forensic signal.
- A larger and more diverse AI donor pool.
- A public leaderboard on the benchmark.
- Signing images at capture on lab imagers.

## Built with

python, opencv, scikit-learn, jax, pymupdf, c2pa, fastapi, react, vite, gemini, synthid, europe-pmc

## "Fresh off the frontier": why this is recent

SynthID Detector opened to the public on October 7, 2026, two weeks before the hackathon. Gemini's current image generation and editing (the Gemini 3 family, including Nano Banana Pro, released November 2025) is also within the 12 month window; the free Gemini app doesn't say which image model served each request, so we describe it as "Gemini's image tools." Both are central: one is the threat, the other is the defense we measured.
