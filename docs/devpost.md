# Devpost submission (final text, ready to paste)

Every number below comes from `results/*.json` in the repo. If a number changes, change it here too.

**Title:** Below the Watermark

**Tagline:** AI watermarks catch whole fake images. We showed they miss a single pasted band, and built a review tool for the figures they can't vouch for.

---

## Inspiration

I work in a wet lab, and Western blots are the evidence behind a huge share of biomedical papers. This year Gemini's image tools made it easy to make a convincing blot, or to change one band in a real one. Google's answer is SynthID, an invisible watermark in every Gemini image, and on October 7, 2026 it opened SynthID Detector to the public. We wanted to know whether that safety net works for science, where image manipulation often happens one band at a time.

## What it does

**The finding.** We tested SynthID Detector by hand, one fixed scenario at a time, within its terms:

- A whole AI blot was detected in 10 of 10 uploads, even after re saving, resizing, JPEG compression, cropping, contrast changes, placement in a figure and a PDF round trip.
- A single AI band on its own came back "not enough details" in 4 of 4.
- An AI band pasted into a published, openly licensed blot came back "SynthID was not detected" in 4 of 4. The untouched published figure, our control, was also not detected, as expected.
- The C2PA content credential Gemini attaches was stripped by the first re save every time.

**The tool.** Below the Watermark is a review assistant for scientific figures. Upload a figure or a whole paper PDF and it:

1. lists every embedded figure by page, and names anything it could not analyze;
2. reads content credentials, and says plainly when one declares an image AI generated;
3. finds the same image used twice in a paper, even after re saving or resizing;
4. screens Western blots band by band with a small neural network we trained, and looks for copied bands;
5. writes a self contained HTML review report with the flagged regions drawn on the figures.

Flags are leads for a human reviewer, never verdicts.

**The benchmark.** An open, labeled benchmark for AI era blot edits: real blot panels from openly licensed (CC BY) papers in Europe PMC, forged five ways (copy move, cross splice, AI splice, intensify, erase), each pushed through realistic publishing steps and saved with a pixel mask of exactly what changed. Existing datasets like BioFors predate generative AI.

## How we built it

- **Data:** CC BY figures from Europe PMC; panels cut out automatically; forgeries generated in Python with OpenCV, scikit image and PyMuPDF, using bands from blots we made with Gemini.
- **Model:** a small CNN written in JAX and trained on CPU. It looks at 32 pixel windows of a blot, normalized against the rest of the same blot, and scores how out of place each one is. Trained on 151 papers and tested on 72 papers it never saw.
- **App:** FastAPI backend, React frontend designed like a film lightbox, `c2pa-python` for credentials, perceptual hashing plus structural similarity for image reuse.
- **Gemini** made our AI blots; **SynthID Detector** was used by hand only.

## Results (honest numbers)

- On 72 held out papers: AUROC 0.703; at our flag line it caught 35% of pasted AI bands while flagging 4% of clean panels.
- End to end, as the app actually runs: we pasted one Gemini band into each of 22 held out published figures. The app flagged the edited panel in 7, flagged no untouched panel in any edited figure, and flagged 1 of the 22 unedited originals.
- After a 50% resize it catches about 5%. Resizing erases most of what it relies on.

That is a first baseline on a hard problem, at a scale where the watermark can't help at all. The benchmark is open so others can beat it.

## Challenges we ran into

- **Resolution.** Web figures put a band in about 13 by 6 pixels. Our first model learned "this is a band" instead of "this band was edited"; hard negatives fixed that. Later we found the opposite problem: publisher PDFs embed figures two to three times larger than what the model learned from, which nearly tripled false flags on clean blots (7% to 18% in a controlled test). The app now scores blots at the training scale.
- **Honesty under a deadline.** Our own first demo figure is a miss: the model does not flag the band we pasted into it. The site shows that with the edit outlined, instead of hiding it.
- **Terms of service.** SynthID Detector forbids automated or adversarial use, so every test was a fixed scenario uploaded by hand, and any adversarial work targets our own model.
- **Data rights.** Every real figure is CC BY and attributed, every edit is ours, and no unpublished lab data is used.

## Accomplishments we're proud of

- A measured, reproducible gap in an AI safety tool that went public two weeks before the hackathon.
- An open benchmark with pixel level ground truth for AI era blot edits.
- A working tool that goes from a paper PDF to a review report in about 20 seconds, with every limit written into the report.
- An independent audit of our own app that found and fixed a crash, a resolution bug behind most false flags, and several overclaims, backed by 29 automated tests.

## What we learned

Watermarks are strong where they are applied (whole images) and absent where science is vulnerable (small regions). Provenance has to be layered, and a detector is only as useful as the honesty of its error rates.

## What's next

- Train on figures at several resolutions and with more AI generators.
- Calibrate flags per paper on unedited published papers.
- Copy move detection inside microscopy images.
- A public leaderboard on the benchmark.
- Signing images at capture on lab imagers.

## Built with

python, jax, opencv, scikit-image, scikit-learn, pymupdf, c2pa, fastapi, react, vite, gemini, synthid, europe-pmc

## "Fresh off the frontier": why this is recent

SynthID Detector opened to the public on October 7, 2026, two weeks before the hackathon. Gemini's current image generation and editing (the Gemini 3 family, including Nano Banana Pro, released November 2025) is also within the 12 month window; the free Gemini app doesn't say which image model served each request, so we say "Gemini's image tools." Both are central: one is the threat, the other is the defense we measured.

## Tracks to enter

Main prizes, Best AI Hack, Best Use of ML (UF DSI), Best Use of Gemini (MLH), Best Social Impact, Best Design.

## Links

- Code: https://github.com/danielsemenovitch/GatorHack
- Video: paste the YouTube or Vimeo link after uploading
