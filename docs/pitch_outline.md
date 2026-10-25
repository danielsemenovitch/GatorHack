# Pitch outline (Monday, about 5 minutes)

**One line:** Below the Watermark is an image integrity checker for scientific papers. It finds what AI watermarks can't: images reused inside a paper, figures lifted from other papers, and pasted-in bands.

**Story:** Google just made AI images checkable, but most image fraud in science never involves a new AI image, so we built the tool that looks below the watermark.

| Time | Beat | Say or show |
|---|---|---|
| 0:00 to 0:30 | Hook | The ALK2 blot with one Gemini band pasted in. "Google's SynthID Detector, public since October 7: not detected." |
| 0:30 to 1:10 | The gap (recent technology) | Whole AI blot detected 10 of 10; one band "not enough details"; a band pasted into a real blot not detected, 4 of 4. "Watermarks protect whole images. Science gets manipulated piece by piece, and much of it isn't AI at all: it's reuse." |
| 1:10 to 1:40 | The problem | Duplication in 3.8% of 20,621 papers (Bik et al., 2016), before generative AI. Paper mills recycle figures across papers. Reviewers check by eye, one paper at a time. |
| 1:40 to 3:10 | Live demo | "A manuscript with two hidden duplications": the blot reused as another protein, boxed on both figures; the figure found in a 2023 paper; then "Search Europe PMC": Gemini reads the manuscript, plans the searches, and the agent finds the original after checking a few papers. Download the report. |
| 3:10 to 3:50 | Numbers | 93% of reused blot panels and micrographs found, even resized, cropped or mirrored; zero false matches across 7,140 pairs of real published figures; every search the agent ran is logged. |
| 3:50 to 4:20 | Proof of concept | "We also trained a CNN for pasted bands inside a single blot: about a third caught at a 4% false alarm rate. With more data and HiPerGator compute, that's the next layer. Our benchmark is open." |
| 4:20 to 5:00 | Vision and close | Journals and reviewers run every submission through it, against the whole open-access literature indexed on HiPerGator. "Watermarks tell you if an image was made by AI. Below the Watermark tells you if a figure can be trusted." |

The deck ("Below the Watermark pitch", Claude Slides) has speaker notes on every slide; the demo slide's screenshot is the fallback if the live demo fails. If the agent can't connect, the library match on the same demo still shows the 2023 source.

## Likely judge questions

- **"Isn't this Proofig or ImageTwin?"** Those are closed commercial tools. Ours is open, searches the open literature with an agent and shows every search it ran, and measures where the new watermark stops.
- **"Does the agent find everything?"** It searches open-access papers on Europe PMC, a few at a time, and logs every query, so a reviewer knows what was and wasn't checked. Indexing the full open-access corpus is the next step.
- **"93% of what?"** Blot panels and micrograph blocks (25 to 55% of a figure) reused in another paper's figure with resizing, cropping, JPEG or mirroring. Single small strips alone: 60 of 120.
- **"Is a match proof of misconduct?"** No. Reuse can be legitimate; the tool shows both images and the source, and a person decides.
- **"Why keep the CNN if it's weak?"** A pasted AI band is the one case nothing else catches. We built the benchmark and a first baseline; the bottleneck is data and compute.
- **"Is anything mocked?"** No. Every number comes from `results/*.json`, produced by scripts in `scripts/`. SynthID results are hand uploads recorded in `results/synthid_manual.json`.
- **"What's recent?"** SynthID Detector (public October 7, 2026) is the defense we measured; Gemini made the AI blots and plans the agent's searches.
