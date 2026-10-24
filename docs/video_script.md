# Demo video script (3 minutes max)

Record the screen at 1080p. One narrator. Numbers in [brackets] come from `results/*.json` and must match the final files exactly.

## 0:00 to 0:25 — The hook

**Screen:** Gemini chat generating a western blot from a one line prompt (one of OUR AI blots), then, if it has been re-recorded on an allowed image, Gemini editing the CC BY ALK2 figure so the FK506 lanes look like the BMP4 lane. **Never** use the old AKT blot recording (see HANDOFF section 2).

**Say:** "Western blots are the evidence behind a huge share of biomedical papers. This year, Gemini made faking one take twenty seconds." Only add "I asked it to make a drug's effect disappear, and it did" if that edit has been re-done on the CC BY figure or our own AI blot, because the original test used private data that cannot be shown or claimed.

## 0:25 to 1:05 — The safety net, and where it stops

**Screen:** synthid.com detecting a whole AI blot. Then the Gap page matrix.

**Say:** "Google's answer is SynthID, an invisible watermark, with a public detector that opened on October 7. We tested it by hand. Whole AI images: caught every time, even after JPEG, resizing and PDF export. The visible content label: gone after a single re-save. But fraud in blots happens one band at a time. Single bands are too small for the watermark to be read, and [pasted band result]."

## 1:05 to 1:40 — What we built

**Screen:** the benchmark folder with a forged image and its mask side by side, then the data card table.

**Say:** "So we built the benchmark this problem needed: [N] blot images from [55] openly licensed papers, forged five ways, including AI bands from Gemini, each pushed through realistic publishing steps, each with a pixel mask of exactly what changed. Then we trained a detector that judges every band against the rest of its own blot."

## 1:40 to 2:30 — Demo

**Screen:** the app. Click "See this example" on the ALK2 figure. Show the heat on the pasted FK506 band, click the band to show the explanation, then click "The same figure, untouched" to show it come back clean.

**Say:** "This is a published, openly licensed figure. We pasted one AI band into the FK506 lanes, which would flip the paper's conclusion. Below the Watermark finds the panels, scores every band, and [result on the example]. It also checks for copied bands, reads content credentials, and hands you the panel to check in SynthID."

## 2:30 to 2:55 — Honest results

**Screen:** the Gap page stats.

**Say:** "On [20] papers it never saw, it reaches an AUROC of [x] and catches [y]% of pasted AI bands at a [z]% false flag rate, including AI bands it never trained on. That's a first baseline on a hard problem, at a scale where the watermark can't even run. The benchmark is open so others can beat it."

## 2:55 to 3:00 — Close

**Say:** "Watermarks protect whole images. Science needs protection one band at a time."

---

**Before recording:** don't show the AKT blot or any unpublished lab data. Every forged image on screen must be one we made from CC BY figures, and say so.
