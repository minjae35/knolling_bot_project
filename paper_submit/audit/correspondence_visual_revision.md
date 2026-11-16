# Object correspondence and failure-margin visual revision — 2026-11-15

This is the latest visual revision record. Earlier Blender and design reviews describe prior stages. The main paper remains five pages; references occupy page 6 and appendices pages 7–12. The final deliverable contains seven figures, six tables, and nine cited research papers.

## Design and evidence

The architecture was rebuilt as a progression of representations: actual overhead RGB, a twelve-channel object-feature matrix, schematic row/pointer structure, and a render of actual saved target poses. A separate lower view exposes context-driven structure selection and the original-feature pose branch. The figure contains no rounded module boxes or fabricated attention values. Primary design references (DINOv2 and Segment Anything) and claim-to-code correspondence are recorded in `architecture_redesign.md` and `design_review_2026.md`.

The README motion demo was inspected. The revised first-page pair uses matched Blender scenes with stable IDs and orientation bars; Figure 4 adds registered source-to-target vectors and an eight-object translation/rotation ledger. Every number is computed from the saved source and target poses. Positive rotation is counterclockwise in world coordinates, with long axes periodic modulo 180 degrees. Arrows connect endpoints and are explicitly not robot trajectories. All original meshes, material colors/roughness, stored poses, cameras, lighting, and render quality are covered by the existing Blender provenance and independent saved-scene audit.

The command-comparison figure now locates failures directly. A separate diagnostic figure combines actual Blender crops, true geometric envelopes, dimensional guides, observed values, and permitted intervals. The full-layout gap is 17.6573727 mm, 0.6573727 mm above its upper limit. Partial clearance is 9.8961028 mm against an effective 9.9000 mm minimum: a 3.8972 µm deficit. This is explicitly a borderline evaluator failure, below native render resolution, not a collision or a claim of micrometer physical accuracy. Ruler scales are labeled and differ between cases.

## Render–inspect–revise checks

1. Composed all figures at their intended 396 pt paper width. Architecture labels are at least 7 pt. Replaced the unsupported ellipsis glyph and corrected the perception-caption alignment; compacted the architecture without reducing type.
2. Compiled the complete manuscript and rendered all twelve pages to PNG in `build/visual_review/correspondence_round1/`. The primary agent and an independent reviewer inspected all pages individually. Title, author, department and university retain their original format; running headers remain absent.
3. Retained the source-panel displacement-vector intersections: stable IDs and the eight-object numerical ledger resolve identity. Do not curve these lines into apparent robot paths or alter target positions to reduce crossings.
4. Extended verification to figure text bounding boxes. Corrected a selector-chart title baseline and added a 2 pt right margin to the displacement-vector note. These are typography changes only.
5. Recompiled and rendered all pages in `build/visual_review/correspondence_final/`. Rechecked the changed pages and all-page overview. No LaTeX warnings, overfull boxes, unresolved references, or text outside page/figure bounds remain.
6. `verify_submission.py` verifies fixed score/evidence/template hashes; eight saved Blender scenes and matched viewing conditions; source-to-target distances and wrapped angles; failure margins; seven PDF/SVG/PNG exports; page counts; and removal of running headers. `package_submission.py` separately extracts, rebuilds and compiles the portable package. Its current result is recorded in `portable_build_check.json`.

The sealed score remains 489/640 → 540/640. The retained staged system, failed broad joint-training pilot, and synthetic-only evaluation are unchanged. No model code, datasets, checkpoints, training processes, saved predictions, or verified experiment outcomes were modified. Legacy figure exports were archived under the excluded local build directory to keep the submission source unambiguous.

Final PDF SHA-256: `af0099de350342e7897b62eb553be656df93031c9b3bd77e49bf8e9af9b9675a`.
