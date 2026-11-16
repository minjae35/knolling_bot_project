# Final submission review — 15 November 2026

Status: **PASS**.

The final PDF is `knolling_bot_v2_paper.pdf`, SHA-256 `ea19c81192c990cb7805d61c63731ab9e269d825f74baf884e2ef67f081f1a1e`. It has five main pages, one reference page, and six appendix pages, with seven figures, six tables, and nine cited papers.

## Scientific review

Independent source and experiment reviews checked the actual adopted loader, perception, language grounding, set context, row/pointer and pose heads, geometry, partial placement, supervision and gradient boundaries. The remote GitHub README and its Blender demo were inspected. Source code governs the more precise account of the text-only 13D goal and separate object-language selection.

The frozen paired scores reproduce 489/640 and 540/640 successes, 51 gains, zero regressions, and the 128-family bootstrap interval [+4.84375, +11.5625] percentage points. The 71 spacing failures are counted uniquely; the four Figure 3 categories partition 100 failures. Dataset/evidence hashes, official-template hashes, and original checkpoint/configuration hashes passed independent checks.

The paper distinguishes staged hybrid inference from the end-to-end learning objective and documents the unsuccessful broader joint pilot. It does not attribute the release gain to one component, claim training-seed confidence, overstate holdout asset novelty, or claim physical execution. Qualitative validation predictions and their incomplete per-record checkpoint provenance are disclosed separately from sealed evaluation.

## Visual review

All 12 pages were directly inspected as full-page renders, and a second independent reviewer also inspected them. The opening teaser is a six-object matched Blender pair; the eight-object movement diagram has its own explicit provenance. Figure captions agree with the final graphics. Overflow pages, a malformed quotation, and stale panel descriptions were corrected. Main text and all figure labels remain inside the page. All PDF fonts are embedded.

The architecture is a reproducible 396 × 232 pt vector figure with labels at least 7 pt, no overlapping labels, and no out-of-bounds text. It shows learned modules, explicit rules, separate goal/selection dependencies, the original-feature path into pose heads, and held/no-op preservation. Matched real Blender meshes and recorded poses support the qualitative results; no generated prediction imagery is used.

Independent Blender checks reopened all ten saved scene files and compared object transforms and every mesh vertex with frozen evidence. The largest absolute pose discrepancy was 5.624e-8, attributable to float32 storage. No layout was changed for presentation; cameras and lighting are matched within each comparison.

## Build and delivery review

`verify_submission.py` passes. The LaTeX log has no unresolved references, missing citations, LaTeX warnings, or overfull boxes. The official ICLR style files remain byte-identical to their upstream versions; this course submission does not claim conference acceptance.

The source package was extracted to a fresh folder without the source repository, original dataset, or model weights. All seven figure PNGs regenerated identically. Independent LaTeX compilation reproduced every page's extracted text and rendered pixels. See `portable_build_check.json` for the machine-readable result.

Original project code, model configuration, datasets, and checkpoints were not changed. Only manuscript, figure, visualization, and audit artifacts were created or revised. Earlier working drafts remain in the parent folder; the reviewed submission is this directory.

The final independent page-by-page review is `final_independent_visual_review.md`; scientific reviews are `implementation_review_20261115.md` and `experiment_review_20261115.md`.
