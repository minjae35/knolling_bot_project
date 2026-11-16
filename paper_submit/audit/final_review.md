# Current canonical manuscript review

Deliverable: `/workspace/knolling_bot_project/paper_submit/knolling_bot_v2_paper.pdf`.

PDF SHA-256: `55a48eb3c9b09f09cdbc417a8dc621089449d3f295dff2025f90bd2601e09aa2`.

The manuscript contains **5 main pages**, 1 references page and 7 appendix pages: 13 pages, 8 figures, 7 tables and 9 cited primary sources. It retains the official ICLR style, existing typography and page settings without conference-submission or acceptance branding.

## Current change: academic prose and narrative

The Abstract, Introduction, Method, training/evaluation, Results, Discussion, Conclusion, Appendix and all figure/table captions were reviewed. Prose was rewritten to connect the motivation, implemented system, experimental evidence and limitations, while reducing telegraphic constructions, rhetorical questions, repetitive caveats and unnecessary punctuation. Figure 6's caption was reviewed and retained exactly as required by its preservation guard.

The Abstract emphasizes the implemented arrangement-prediction system and verified independent-set performance. It describes the fixed learned base and synthetic scope. The failed broader joint-training pilot is documented in the training section and Appendix B, distinct from the adopted restricted joint fine-tuning. No unsuccessful experiment is presented as successful, and no claim of physical execution or fully differentiable end-to-end training is introduced.

The Discussion connects the unchanged Figure 4 motion measurements to complete-scene validity. Appendix D now focuses on command-dependent examples, while Appendix G consolidates the eight-object coordinates, existing rotation equation, row memberships and spacing measurements. Conclusions distinguish future joint-training research from physical validation without making one a prerequisite for the other.

## Scientific and asset preservation

`prose_revision_review.json` records all 139 protected file hashes, which match the previous submission. Figure exports, editable figure masters, Blender renders/scenes, stored poses, experimental logs and scientific audit data remain unchanged. All eight display equations, seven table bodies and eight figure inclusion commands are byte-identical. The bibliography and cited sources remain unchanged, and the references page is pixel-identical. Figure 6 matches its frozen exports, four renders, four Blender scenes and complete LaTeX block.

An independent semantic review checked architecture, inference routing, gradient boundaries, training stages, evaluation splits, paired comparisons, bootstrap interpretation, qualitative provenance and physical-execution limitations. Two potentially overbroad phrases were corrected: held/no-op poses are preserved separately from mover refinement, and the color-regularizer statement describes the available loss records without inferring the missing trainer's complete objective. A final review confirmed preservation after consolidating Appendix G.

The result remains 489/640 → 540/640, with 51 paired gains and no regressions. The family-bootstrap interval remains [4.84375, 11.5625] percentage points. Discussion values for objects 1 and 4 repeat unchanged Figure 4 values. The full timing evidence remains in Appendix F. No model inference, training, dataset changes or checkpoint changes occurred.

## Full-manuscript review and compilation

All 13 pages were rendered and inspected, including detailed checks of dense methods/training pages and the final Appendix D/G arrangement. The revised prose and captions fit without changing fonts, margins, spacing settings or figure sizes. The Conclusion remains on page 5, References on page 6, and Appendix A begins on page 7. There are no LaTeX warnings, overfull boxes or unresolved references.

`submission_checks.json` verifies page structure, scientific evidence, all figure bounds, frozen Figure 6, paired-bootstrap results, object-motion directions/measurements and every Table 7 value. Preview images are in `build/visual_review/prose_final/` and are excluded from the submission ZIP.

## Source package

`portable_build_check.json` records the extracted-source rebuild and checks all 13 pages by text and rasterized pixels. The existing packaging check regenerates figures only in its isolated temporary copy, verifies their pixels against the unchanged canonical artwork, and never alters the canonical figure files. The ZIP includes revised LaTeX, bibliography, appendix, final PDF, editable figure sources, approved Blender scenes/renders, evidence and build scripts. It reproduces the paper and its analyses, not the original training process.

Previous visual revision records, including `figure4_motion_review.json`, `figure7_grid_review.json`, `figure7_zoom_review.json`, `figure7_scale_review.json` and `design_reference_review.json`, remain unchanged and describe the artwork retained in this prose revision.
