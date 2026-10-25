# Note to the SynthID team (ready to send)

**To:** the SynthID feedback channel (the feedback icon at the top right of synthid.com), and Google's Vulnerability Reward Program intake if you want a tracked report: https://bughunters.google.com
**Subject:** SynthID Detector and band level edits in scientific images (student hackathon finding)

Hi SynthID team,

I'm an undergraduate at the University of Florida. This weekend at Gator Hack V (UF AI Days) my team tested the public SynthID Detector on western blot images, the kind used as evidence in biomedical papers. Everything was uploaded by hand, one fixed scenario at a time, following the detector's terms.

What we saw:

1. Whole images from Gemini's image tools were detected every time, including after re-saving, resizing to 50%, JPEG at quality 60, cropping, contrast changes, placement as one panel of a figure, and a PDF round trip. Impressive robustness.
2. Tight crops of a single band (roughly 80 to 200 pixels wide) returned "not enough details."
3. A band cut from a Gemini generated blot and pasted into an openly licensed (CC BY) published blot returned "SynthID was not detected" in 4 of 4 cases (four different published figures, saved as JPEG), while whole Gemini blots were detected every time. The untouched published figure, our control, also returned "SynthID was not detected," as expected.

Point 3 matters for research integrity, because image manipulation in papers often happens at the scale of a single band. We are not publishing a step by step recipe. Our public materials describe the gap only at the level above, and we are releasing a band level forensics tool and benchmark to help cover it.

The composites were made by us from openly licensed (CC BY) figures; the published figures and their authors have nothing to do with the edits. Happy to share the exact files privately if useful.

Thanks,
Daniel Semenovitch
University of Florida
