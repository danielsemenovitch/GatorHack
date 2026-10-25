# Methods and judge's guide — current submission

## Purpose

Below the Watermark is a **scientific-image duplication review assistant**. It helps a human compare raster figures in a paper, find image regions reused in other figures, and trace potential reuse to open-access published sources. It does **not** determine research misconduct or publication priority.

## 1. Figure inventory and provenance

Upload a JPEG, PNG or PDF (25 MB maximum). PyMuPDF extracts embedded raster figures; the report identifies skipped/unreadable content. C2PA credentials are verified where available for supported original image files. The app **does not** check individual image signatures inside PDFs. SynthID has no public analysis API in this project; all SynthID case-study tests were done manually.

The manual SynthID observations are recorded in `results/synthid_manual.json`: 10/10 whole Gemini blot variants detected, 4/4 pasted-band composites not detected, and isolated single-band inputs too small to assess. This is a **small case study**, not a general SynthID performance estimate. A draft disclosure to Google's team is in `docs/disclosure_email.md`; external submission is **not confirmed**.

## 2. Duplication matching (the submission's central feature)

**Whole image:** decoded image pixels and perceptual similarity/structural checks identify exact or near duplicates.

**Region-level reuse across separate figures:** `btw/region_match.py` extracts SIFT descriptors from visual content, suppresses mostly text/blank areas, uses one-to-one feature matching and geometric RANSAC alignment, then requires at least 0.90 aligned content correlation. It can tolerate resizing, JPEG changes, mirroring and some cropping. It does *not* scan for copying **within one composite figure**, and small texture-poor strips can be missed.

**Across papers:** a bundled reference library contains **112 CC BY figures from 62 papers** (demo-source exclusions are intentional). The optional `Search Europe PMC` action plans related-paper queries using Gemini if `GEMINI_API_KEY` is configured, or deterministic keywords otherwise. It retrieves open-access paper figures and runs the same matcher. Queries, paper IDs, counts and errors are reported; its result is bounded to the papers actually checked. A successful local-library match does not mean the web agent ran.

**Controlled region benchmark (PR #8):**

| Test class | Detected | Why both matter |
|---|---:|---|
| **Larger reused blot/micrograph blocks (~25–55% of a source figure)** | **110/118 (93.2%)** | Resized, cropped, mirrored or JPEG-compressed source blocks pasted into different figures. `results/region_match_eval_blocks.json` |
| **Single smaller reused panels, often narrow blot strips** | **60/120 (50%)** | Lower texture and less spatial structure. `results/region_match_eval.json` |
| **Untouched reference-image pairs** | **0/7,140 false matches** | Tested pairs in the provided library only, not an independently sampled real-world false-positive rate. `results/region_match_eval.json` |

Both positive benchmarks are artificially constructed from openly licensed published figures; neither establishes recall on authentic forensic casework. The online agent's search-retrieval accuracy has **not** been benchmarked at 93.2%.

Reproduce: `python -m scripts.eval_region_match --blocks`, `python -m scripts.eval_region_match`, and `python -m scripts.eval_region_match --fp`. The last option checks all untouched reference pairs and takes longer.

## 3. Experimental Western blot CNN (supporting proof of concept)

The CNN in `models/splice_cnn_v3.npz` uses JAX and sliding 32-pixel patches, with intensity and noise features relative to the rest of each blot. The team made synthetic edits: copied bands, cross-spliced bands, AI-donor splices, intensity changes and erasures. Panel scores are **review indices, not probabilities**. Standard flag line is 0.50 and Strict line is 0.60, calibrated separately.

On 2,761 synthetic test images from 72 held-out papers, v3 measured AUROC 0.703, detected 35.4% of AI-splice panels and flagged 4.1% of clean panels (`results/cnn_v3_metrics.json`). The full application flagged an edited panel in 7 of 22 artificially edited whole figures (`results/figure_level_eval.json`). After a 50% resize, recall was about 5%. **Most edits go undetected**. V4/V4.1 research runs are isolated, and no model should be promoted solely based on training loss.

The separate copied-band noise matcher has low recall and should be treated as supplementary.

## 4. Review, attribution, downloadable evidence

All candidate regions are presented side by side. The downloaded standalone HTML report contains performed/skipped checks, matched crops for within-paper and local-reference matches, source IDs and links, and—**only if requested before export**—the on-demand Europe PMC query/paper log and matches. A missing source or an unexecuted web search is not silently treated as “searched, nothing found.”

The demonstration PDF `examples/demo_reuse.pdf` contains **two team-created duplications** and explicitly identifies CC BY original figures and alterations. Original authors are not implicated.

## Questions judges may ask

- **“93% of what?”** Only our specific benchmark of larger artificially reused blocks; tiny strips were 50% and other modalities remain unvalidated.
- **“Did you search the whole scientific literature?”** No. The bundled 112-image library is finite; the optional agent searches a small on-demand set of related Europe PMC open-access papers.
- **“What happens if Gemini isn't configured?”** Search falls back to keyword planning. The matcher itself uses classical computer vision, not Gemini.
- **“Does duplication mean fraud?”** No. Legitimate reuse exists; reviewer must read captions, citations and methods.
- **“How good is the AI-edited-band model?”** Weak, experimental and very resolution-sensitive; it is a proof of concept, not our headline tool.
- **“Can we verify these claims?”** Yes. Figures have source metadata, controlled edit scripts are in `scripts/`, numeric outputs in `results/`, and backend/frontend checks run in CI. Internet searches require a separate live network check.
