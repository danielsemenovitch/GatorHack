# Devpost submission (final text, ready to paste)

Every number below comes from a file in `results/` or a recorded run. Places marked **[V5]** change if the V5 model is promoted; leave them as written if it is not.

**Title:** Below the Watermark

**Tagline:** An image integrity checker for scientific papers. It finds what AI watermarks can't: images reused inside a paper, figures lifted from other papers, and pasted-in bands.

---

## Inspiration

I work in a wet lab, and Western blots are the evidence behind a huge share of biomedical papers. Image duplication turned up in 3.8% of 20,621 papers even before generative AI (Bik et al., 2016), and paper mills recycle figures across papers. Now Gemini's image tools make a convincing blot in seconds. Google's answer is SynthID, an invisible watermark in every Gemini image, and on October 7, 2026 it opened SynthID Detector to the public. We wanted to know whether that new safety net protects science.

## What it does

**The finding that started it.** We tested SynthID Detector by hand, one fixed scenario at a time, within its terms:

- A whole AI blot: detected in 10 of 10 uploads, even after re-saving, resizing, JPEG, cropping, contrast changes, placement in a figure and a PDF round trip.
- One AI band on its own: "not enough details", 4 of 4.
- One AI band pasted into a published, openly licensed blot: "not detected", 4 of 4.
- The C2PA content credential Gemini attaches was gone after the first re-save, every time.

Watermarks protect whole AI images. Science gets manipulated one figure, one panel, one band at a time, and the oldest trick, reusing an image, involves no new AI image at all.

**The tool.** Upload a figure or a whole paper PDF. Below the Watermark:

1. **Finds images reused inside the paper.** Every pair of figures is compared region by region, so a blot reused under another label is found even when it was cropped, resized, re-saved or mirrored.
2. **Finds figures lifted from other papers.** Every figure is checked against a library of published figures, and a **search agent** goes further on request: Gemini reads the manuscript's title, abstract and captions and plans Europe PMC searches for the same authors and the same topic, the agent downloads the figures of the open-access papers it finds, and the same matcher checks each one. Every query and paper it checked is listed.
3. **Reads content credentials** and says plainly when one declares an image AI generated; hands off to SynthID Detector for the manual check.
4. **Screens Western blots band by band** with a small neural network we trained (a proof of concept, see below).
5. **Writes a self-contained HTML evidence report**: matched regions side by side with source citations and links, flagged panels, and everything it checked or skipped.

Flags are leads for a human reviewer, never verdicts: reuse can be legitimate.

## How we built it

- **Duplication matcher:** SIFT keypoints on image content only (text on white is masked out, since labels like "beta-actin" use the same fonts in every paper), one-to-one matching, a RANSAC similarity transform, and a pixel check on the aligned regions, so a cropped, resized or mirrored copy still lines up but a chart drawn from the same template does not. The same matcher runs within a paper, against a library of 112 published CC BY figures from 62 papers, and on every figure the agent downloads.
- **Search agent:** Gemini (`gemini-flash-latest`) plans the Europe PMC queries from the manuscript text, or reads the protein labels off a single figure; keyword rules take over if Gemini is unavailable. The Europe PMC REST API supplies open-access papers and their figures.
- **Blot CNN:** a small JAX network that scores 32-pixel windows of a blot against the rest of the same blot, trained on our own benchmark: real blot panels from CC BY papers, edited five ways (copy-move, cross-splice, AI splice, intensify, erase) with bands from blots we made with Gemini, pushed through realistic publishing steps, each with a pixel mask.
- **App:** FastAPI backend, React frontend designed like a film lightbox, `c2pa-python` for credentials, PyMuPDF for PDFs.
- **Gemini** made our AI blots and plans the agent's searches; **SynthID Detector** was used by hand only.

## Results

**Duplication (the headline):**
- A blot panel or micrograph block (25 to 55% of a published figure) reused in a figure from another paper, resized, re-saved, cropped or mirrored: **110 of 118 found (93%)**. Single small strips alone are harder: 60 of 120.
- **0 false matches** across 7,140 pairs of untouched published figures.
- **Live agent run:** on our demo manuscript, Gemini planned the searches and the agent found the 2023 source of a lifted figure after checking **4 papers** (keyword planning needed 16 papers and 55 figures).

**Blot CNN (proof of concept) [V5]:** on 72 held-out papers it caught 35% of pasted AI bands while flagging 4% of clean panels (AUROC 0.703). End to end on 22 published figures with one pasted Gemini band, it flagged the edited panel in 7 and 1 of 22 unedited originals. After a 50% resize it catches about 5%. A useful baseline, not a finished detector: the bottleneck is training data and compute.

These are controlled tests of duplicated or edited pixels, not measurements of real-world misconduct.

## Challenges we ran into

- **Watermarks and small regions.** SynthID can't judge a single band, and nobody had measured that for science; we did it by hand, within the detector's terms, with no automated probing.
- **Telling copies from look-alikes.** Figures in one paper share chart templates and layouts; our first matcher linked them. A pixel check after alignment fixed it without losing real copies.
- **Busy figures.** A cap on keypoints let the rest of a figure crowd out a reused panel; removing it took reused-block detection from about 70% to 93% with no new false matches.
- **Resolution.** Publisher PDFs embed figures two to three times larger than the web figures the CNN learned from, which nearly tripled false flags; the app now scores blots at the training scale.
- **Data rights.** Every figure is CC BY and attributed, every edit is ours, and no unpublished lab data is used.

## Accomplishments we're proud of

- A measured gap in an AI safety tool that went public two weeks before the hackathon.
- A search agent that finds a figure in other published papers, live, and shows its work.
- 93% of reused panels found with zero false matches on 7,140 real figure pairs.
- An open benchmark with pixel-level ground truth for AI-era blot edits.

## What we learned

Watermarks are strong where they are applied (whole images) and absent where science is vulnerable (small regions and reused images). Integrity checking has to be layered, and a detector is only as useful as the honesty of its error rates.

## What's next

- Index the full open-access literature on HiPerGator, so every upload is checked against millions of figures instead of 112.
- Train the blot model on far more data and more AI generators with HiPerGator compute.
- Copy-move detection inside a single figure.
- Integrate with journal submission and peer review systems.

## Built with

python, jax, opencv, scikit-image, scikit-learn, pymupdf, c2pa, fastapi, react, vite, gemini, gemini-api, synthid, europe-pmc

## "Fresh off the frontier": why this is recent

SynthID Detector opened to the public on October 7, 2026, two weeks before the hackathon, and it is the defense we measured. Gemini's current models are central twice: its image tools (the Gemini 3 family, including Nano Banana Pro, released November 2025) made the AI blots, and `gemini-flash-latest` plans the agent's literature searches.

## Tracks to enter

Main prizes, Best AI Hack, Best Use of ML (UF DSI), Best Use of Gemini (MLH), Best Social Impact, Best Design.

## Links

- Code: https://github.com/danielsemenovitch/GatorHack
- Video: paste the YouTube or Vimeo link after uploading
