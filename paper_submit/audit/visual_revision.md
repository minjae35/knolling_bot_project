# Visual redesign review — 15 November 2026

This revision addresses the author's request for a substantial redesign of the architecture, qualitative examples, quantitative plots, and overall typography. All edits are confined to `paper_submit/`; model code, datasets, checkpoints, and frozen score records are unchanged.

## Design and evidence

- Rebuilt the architecture as three aligned stages: perception/grounding, learned layout prediction, and constrained decoding. Original relation features bypass set attention for coordinate heads; contextual features guide row/order selection. Held/no-op estimates have a separate preservation route. The duplicate filter is explicitly marked as a rule. This is an inference overview, not a claim of joint differentiable training.
- Rebuilt the main qualitative figure around the actual input, source-to-target displacement vectors, and the existing render at stored predicted poses. The requested right region is overlaid consistently. No RGB synthesis or new model inference was used. Displacement arrows do not represent collision-free robot trajectories.
- Gave the shared-image qualitative diagnostic a dedicated appendix page. Full placement, its failed-gap enlargement, blue-only placement, and preservation use stored predictions with true footprints/envelopes solely for visualization and scoring.
- Recomputed geometric annotations from the saved poses and source dimensions, without rescoring the sealed evaluation. Full-layout row gaps are 15.69957 and 17.65737 mm; the latter exceeds 15 ± 2 mm. Partial clearance is 9.89610 mm between o06 and o08, narrowly below the effective 9.900 mm boundary (nominal 10 mm minus 0.1 mm numerical tolerance). No-op maximum drift is 0.41739 mm and 0.29237 degrees. Inputs and exact values are recorded in `visual_measurements.json`.
- Verified envelope rotation, Euclidean clearance, row inference, and numerical tolerance against `src/knolling_dataset/validator.py` and `src/knolling_model/evaluation.py`. This is a visualization of known failures, not new empirical performance evidence.
- Replaced the error plot with paired dot comparisons, using the same saved counts; retained zero-based rate axes and marked overlapping error categories. The training plots distinguish component accuracy from complete-scene success.
- Unified five figures at the manuscript's 5.5-inch width, with blue/teal accents, consistent sans-serif labels, restrained gridlines, panel labels, and direct annotations. Exports include vector PDF, editable SVG text, and 300-dpi PNG. Bundled Liberation Sans TrueType fonts and their license make the typography reproducible.
- Updated table widths, header emphasis, row spacing, and a continuous overall-result highlight. Bibliography text is ragged right to eliminate stretched word spacing. Main-body technical prose was tightened to retain five pages, with exact implementation details in the appendix.

## Render–inspect–revise cycle

1. Preserved the original manuscript and figure generator under `build/visual_before/`.
2. Generated redesigned figures and compiled an 11-page first revision. Rendered every page at 1.6× resolution and visually inspected all 11 page images. Corrected crowded block labels, quantitative annotation overlap, source-thumbnail spacing, zoom bounds, and the discontinuous table highlight.
3. Recompiled and rendered all pages again. Improved table/bibliography typography and inspected the changed pages.
4. PDF font inspection revealed an OpenType embedding compatibility warning. Replaced the figure font with a bundled metrically similar TrueType family, regenerated all figures, and recompiled. `pdffonts` now reports embedded fonts with no warnings. No Type 3 fonts are present.
5. Rendered all 11 final pages and inspected all overview sheets plus detailed architecture and qualitative pages. Final PDF: main text 1–5, references 6, appendix 7–11; five figures and six tables. The submission verifier passes, including original evidence/template hashes and absence of overfull boxes or unresolved references.

Rendered review images are in `build/visual_review/{round1,round2,final}/`; `scripts/render_pages.py` reproduces them. The portable-package check is recorded separately in `portable_build_check.json`. This review assesses the actual document; it does not promise a course grade.
