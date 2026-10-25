# Demo video script (under 3 minutes)

**Setup (5 minutes before recording):** start the app (README "Run it", then open http://localhost:8000), browser window about 1440 px wide, zoom 100%, close other tabs. Record the screen at 1080p with your voice (OBS, or Windows Game Bar with Win+Alt+R). Read the lines below at a calm pace; they are about 400 words, which fits in 2:45.

Never show the old AKT blot or any unpublished lab data. Every edited figure on screen is one we made from a CC BY figure, and the site says so.

---

**0:00 to 0:20. The hook.** Screen: the landing page, hero visible.

> Western blots are the evidence behind a huge share of biomedical papers. This year, Gemini made a convincing fake blot a twenty second job. Google's safety net is SynthID, an invisible watermark in every Gemini image, with a public detector that opened on October 7. We tested it on science.

**0:20 to 0:55. The finding.** Screen: click "The gap". Hold on the three part summary, then scroll slowly through the table.

> We uploaded every test by hand. A whole AI blot: detected ten times out of ten, even after JPEG, resizing, cropping and PDF export. One AI band on its own: not enough details. And the case that matters for science: one Gemini band pasted into a real published blot. SynthID did not detect it, four times out of four. The content credential was gone after the first re save. So the watermark protects whole images, and science gets edited one band at a time.

**0:55 to 1:15. What we built.** Screen: click "How it works", scroll through the five steps.

> So we built Below the Watermark, a review assistant for the figures a watermark can't vouch for. Drop in a figure or a whole paper PDF. It lists every figure, reads content credentials, finds images used twice, screens Western blots band by band with a model we trained, and writes a report a reviewer can check.

**1:15 to 1:35. Provenance.** Screen: click "Review images", then "A whole blot made by Gemini".

> When the credential is still there, we read it. This blot came straight from Gemini, and the report says so up front.

**1:35 to 2:10. The detector, honestly.** Screen: click "Another AI band the model catches (selected)". Point at the dashed box and the flagged panel. Then scroll up and click "Run our checks on this blot" (the ALK2 figure) and point at the dashed box.

> This is a published figure where we pasted one AI band into the IL 6 strip. The dashed box is our edit, and the model flagged the panel that contains it. Now our first example: same kind of edit, and the model misses it. We show both, because both happen. On twenty two held out figures, it flagged the edited panel in seven, and it flagged one of the twenty two unedited originals.

**2:10 to 2:35. A real paper.** Screen: drag the Oncotarget PDF onto the page (or have it already analyzed), scroll the report, then click "Download review report".

> It runs on whole papers. This PDF takes about twenty seconds: thirty blot panels, two weak flags, and the report says about that many are expected by chance. Everything it did and everything it skipped goes into one HTML report.

**2:35 to 2:50. Close.** Screen: back to the landing page.

> Our benchmark of AI era blot edits is open, so others can beat our model. Watermarks protect whole images. Science needs protection one band at a time.

---

**If the Oncotarget PDF isn't handy (or you'd rather show reuse):** at 2:10 click "A demo PDF with one figure used twice" instead and say: "It runs on whole papers. In this PDF we put the same figure in twice, once resized and recompressed. The report puts the pair at the top, screens every blot panel, and everything goes into one HTML report."
**If you run over:** cut the provenance section (1:15 to 1:35) first.
**Upload:** YouTube (unlisted is fine) or Vimeo, then paste the link into Devpost.
