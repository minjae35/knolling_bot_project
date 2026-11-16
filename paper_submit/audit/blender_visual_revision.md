# Blender publication redesign — 15 November 2026

## Reference and narrative decisions

Inspected the actual README representative demo `docs/v2.0.0/demo-gallery/motion/13_full_pass.png` and the project Blender worker. An independent reviewer inspected primary DINOv2 and Segment Anything papers; links and actionable design principles are preserved in `design_review_2026.md`. No external figures or experimental results were copied.

The visual narrative now starts with a matched 3D before/after example on page 1, then separates an inference overview from the learned layout subgraph on page 2. The original-feature coordinate branch, contextual selector branch, object-conditioned goal, and held/no-op bypass correspond to actual implementation. Page 5 compares three instructions on a common source scene. Appendix D explains failures using real Blender closeups with envelope measurements and separately shows the actual model input images.

## Blender fidelity and output

Eight new 1920×1280 renders: six oblique views and two calibrated overhead views. Blender 5.1.0, Cycles CPU, eight threads, adaptive sampling with 128-sample maximum, denoising. Original footprint extrusion, heights, palette and roughness are preserved. Presentation changes are camera, illumination, neutral tabletop and rendering quality. No object-layout adjustment, new inference, training or sealed evaluation was performed.

`blender_provenance.json` records source/output hashes, cameras, lights and input poses. `blender_pose_checks.json` independently opens all eight saved `.blend` files and compares every object transform and mesh vertex with frozen evidence. All pass at absolute tolerance 1e-7; the largest pose discrepancy is 5.6233e-8, attributable to Blender float32 storage. The submission verifier also checks identical cameras, lights, materials, resolution and sampling settings within each comparison group.

The partial-failure overlay measures oriented bounding envelopes, not the visible chamfer/capsule silhouettes. Dashed envelope outlines clarify the 9.896 mm clearance; the effective lower bound is 9.900 mm. The full failure's measured row gap is 17.657 mm against 15 ± 2 mm. No-op max drift remains 0.417 mm / 0.292 degrees. These are read-only measurements of saved validation poses, not a new benchmark.

## Page review and revision

- Preserved the preceding manuscript and generator under `build/pre_blender_redesign/`.
- Constructed paper-size layout proofs while the final Cycles batch ran; these temporary proofs are confined to `build/proof/` and are not submission assets. Resolved section spillover by shortening repeated prose, compacting the abstract/related-work treatment, and retaining implementation depth in appendices.
- Checked the actual new architecture and matched Blender pair at publication size. Fixed lower-module text padding, raised diagram labels to at least 7 points, and explicitly separated actual inputs from presentation renders.
- Compiled the complete document with all finished render assets. Rendered all eleven pages to `build/visual_review/blender_final/` and inspected every page. Refined the long provenance paragraph to avoid stretched spacing and removed a short orphan line in Appendix A.
- Recompiled and rendered all eleven pages to `build/visual_review/blender_final2/`. Checked the revised pages and the complete document: main text 1–5, references 6, appendix 7–11; six figures, six tables and nine cited primary sources. No unresolved references, overfull boxes, or PDF font warnings remain.

All primary counts and frozen evidence hashes are unchanged. The original project code, datasets and checkpoints are untouched. `submission_checks.json` reports the final artifact's integrity checks; `portable_build_check.json` records isolated source-package recomposition/compilation. This review assesses the actual document without predicting a course grade.

## Author-requested title restoration

Restored the pre-redesign `\maketitle` block: both title lines bold, colon after v2, author name followed by department and university on separate lines. Adjusted only the first figure’s surrounding whitespace to retain the five-page main text. First page rendered and inspected; pages 2–11 retain identical extracted text. Updated PDF and portable source package.

## Running-header removal

Removed the added gray running title/author headers at the author’s request. The restored title/author block and centered page numbers remain.
