# Independent visual review of submitted manuscript snapshot

Reviewed the entire PDF visually at 1.6x resolution and overview scale. Snapshot SHA-256: `7057cf3e45a908488b82f8e287d47eda8511c88c29836fb4a12a8e39650c8461`. The snapshot, per-page PNGs, overview sheets, and extracted text are preserved in this directory. Original paper files were not changed.

## Required before submission

1. **Restore the requested five-page body.** This snapshot is 14 pages, with six body pages. Page 2 is a nearly empty continuation of the Introduction, forced by an oversized opening teaser and the `clearpage` before Method (`main.tex:35–52`). Removing teaser panel (c), which duplicates the richer motion figure, should recover the full Introduction on page 1 while keeping the actual Blender before/after pair. Do not reduce body typography to fix this.
2. **Eliminate the appendix orphan page.** Appendix A spills about 10 lines onto page 9, which is otherwise blank (`appendix.tex:27–44`). Condense the implementation table and explanatory text, move the spacing equation to the main Method, or allow Appendix B to follow without a page break. A coherent one-page Appendix A is preferable.
3. **Fix the Figure 3 caption.** The actual graphic now shows (a) paired outcomes, (b) complexity, and (c) four mutually exclusive spacing categories. Its caption describes an obsolete two-panel graph and seven overlapping error categories (`main.tex`, Figure 3 caption). The current caption is scientifically misleading. Suggested text: “Saved-score diagnostics. (a) Every v1 success is retained, with 51 additional successful scenes. (b) Success by object count; bins contain 90 or 95 pairs. (c) The 100 v2 failures partition into horizontal-only, both-gap, vertical-only, and neither-gap groups; 71 have a spacing violation.”
4. **Fix malformed quotation.** `main.tex:55` has an unmatched opening quote before “Vertical”; use “The vertical-axis command rotates objects; rows still run left to right.”
5. **Align example provenance.** Current Figure 1 shows eight objects, but Appendix D calls it a six-object opening example and cites a different scene. Keep the selected figure, caption, and provenance scene ID consistent after replacement.
6. **Regenerate package and checks.** README says 12 pages, while current PDF has 14. The ZIP and existing PASS report predate the current PDF. `verify_submission.py` validates `architecture_new` even though LaTeX includes `architecture_main`. Align assets, report counts, PDF hash, and package after the final build.

## Page-by-page assessment

| Snapshot page | Assessment and action |
|---|---|
| 1 | Title/author typography is professional. Eight-object Blender pair is readable. Motion panel is redundant and too tall; it causes severe overflow. Remove that panel and retain the full motion ledger later. Add the actual command or compact target intent to the teaser if available. |
| 2 | Unacceptable short orphan page with the remaining Introduction and Related Work. Must merge into page 1. |
| 3 | Architecture is substantially clearer than a generic box diagram; original versus contextual feature routing is visible. Small labels remain readable at actual print size. Upper diagram still visually conflates text-only global goal with object-conditioned selection; split their labels/routes if revising. Correct the unmatched quote. |
| 4 | Dense but well-structured data/training/protocol page; table is readable and does not visibly clip. Strong distinction between corpus scale and actual stage exposure. |
| 5 | Attractive quantitative figure; table and figure reinforce the paired comparison. Caption is materially stale. Spacing chart is a mutually exclusive partition, not overlapping categories. |
| 6 | Motion ledger is useful and readable; translations and signed periodic rotations are explicit. This is the appropriate location for detailed correspondence. Adequate whitespace below the conclusion. |
| 7 | References are legible and well spaced; long URLs wrap within bounds. No visible clipping. |
| 8 | Detailed implementation table and equations are legible. Too much text for one page, causing the orphan on page 9. |
| 9 | Unacceptable mostly blank appendix overflow page. |
| 10 | Training evidence, loss definitions, gradient boundaries, and historical limitations are well presented. No obvious layout defects. |
| 11 | Learning-curve plots are clear and appropriately labeled as historical diagnostics. Data/statistical details and error table fit. Bottom table approaches normal text-area bottom but is not clipped. |
| 12 | Three-command Blender matrix is consistent and readable. The 3.9 micrometre deficit is explicitly described as below image resolution, avoiding visual overclaim. Update opening-example provenance. |
| 13 | Strong failure visual: calibrated envelopes and numerical interval rulers explain hidden threshold violations. Distinguishes envelope clearance from mesh silhouette and scoring failure from collision. |
| 14 | Reproducibility page is readable. Update paths and package assertions if final revision is relocated. Table count reference is hard-coded. |

## Design recommendations

- Keep the current restrained navy/teal palette, vector architecture/plots, and real Blender geometry.
- Use a compact before/after teaser plus one full motion figure; avoid repeating the same correspondence visualization twice.
- Preserve the exact distinction between image-to-pose inference and end-to-end optimization. An explicit small factorization or spacing equation on the Method page will improve technical depth more than another decorative figure.
- Main figure labels are mostly readable; do not shrink them to compensate for pagination.
- The gray Blender surfaces are somewhat flat but scientifically serviceable. A stronger studio render is optional; exact poses, camera consistency, source meshes, and command clarity are more important.
- Reinspect every page from the final PDF after all changes. Do not rely on this snapshot or earlier PASS files.
