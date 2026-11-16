# Visual narrative and architecture review — 2026-11-15

Read-only review of the manuscript, README demo, and model implementation. This note supplies design guidance; it does not assert that every suggestion has already been implemented.

## Primary design references inspected

- [DINOv2, primary paper](https://arxiv.org/pdf/2304.07193), rendered pages 2 and 4, Figures 1 and 3. Figure 1 places corresponding images/results in tightly registered pairs. Figure 3 makes data visible through actual thumbnails and makes processing legible through a single left-to-right axis, sparse stage separators, and labels near operations. It is not a wall of equal-sized prose boxes. For this project, keep before/after camera, crop, and object appearance identical; make image → object representation → structure → geometry → rendered prediction the visual backbone. Borrow hierarchy and alignment, not their data or artwork.
- [Segment Anything, primary paper](https://arxiv.org/abs/2304.02643), [PDF](https://arxiv.org/pdf/2304.02643), rendered pages 1, 3, and 5, Figures 1, 2, and 4. Its opening overview precedes the abstract; the detailed model overview is a separate, compact figure. Actual input and output thumbnails anchor an otherwise abstract diagram. Large visual groups communicate task, model, and data before readers inspect details. For this manuscript, a matched Blender before/after teaser can establish the task immediately, while the next page handles module routing. A small-multiple qualitative grid should use constant framing and put outcome labels in the same position.

These are design references, not additional related-method claims or experimental baselines. No copied artwork is needed.

## Why the preceding revision still felt like a report

The old architecture gave every noun a similarly weighted rectangle. Numerous internal labels and crossing routes forced readers to reconstruct the story. The successful arrangement appeared only near the end of the main text, and intermediate 2D shapes received as much space as the actual render. The appendix showed predicted objects as flat polygons despite the project's available 3D assets. Improved colors and typography could not resolve that narrative mismatch.

The README's `docs/v2.0.0/demo-gallery/motion/13_full_pass.png` was inspected directly. Its registered initial/predicted scenes and explicit command demonstrate the right comparison, but the central motion panel, object-number markers, and angle list are too dense to reproduce wholesale at a 5.5-inch paper width. The raw 3D before/after pair should be larger; movement can be supplementary rather than competing with it.

## Recommended architecture at 396 × 200 points

Use two levels rather than three tall columns:

1. An approximately 80-point overview: real overhead RGB and a compact English-input strip → object and goal representations → layout proposal → constraint decoding → actual target-render thumbnail. Labels sit below visual representations. Use at most two shaded group containers. Token/row symbols are schematic, explicitly not measured activations.
2. An approximately 85-point enlarged layout detail: relation features `h` split into a contextual selector branch and an original-feature coordinate branch. The contextual branch takes goal `g` and size, applies one four-head attention block, then predicts row membership and pointer order. Coordinate heads receive original `h`, `g`, and the discrete structure. Their outputs join the constraint stage. Keep geometry operations as concise verb labels rather than five stacked boxes.
3. A lower bypass for held/no-op objects: detected centers plus RGB/PCA orientation correction → output merge. Partial mode sends only the selected subset through arrangement, then applies held-object clearance. This routing must not suggest every object always moves.

Keep the overall flow horizontal. Prefer broad alignment and whitespace to tiny text. At print size, target approximately 8-point labels, with no label below 7 points unless it is optional metadata. Do not add transformer stacks or attention heat maps that imply unavailable architecture depth or recorded attention values.

## Code correspondence checked

- `src/knolling_model/experiments/size_refinement.py::poses_from_detection`: perception; optional size refinement; duplicate suppression; object packing; MiniLM instruction encoding; object-conditioned goal encoding; explicit region/gap alignment; full versus held routing. The diagram must show that object information reaches grounding, not only the arrangement model.
- The full route calls relation hidden features, row model, candidate selection, area order, geometry, infeasible fallback, row repair, and color separation. These are not one newly trained end-to-end block.
- `set_context.py::ObjectSetContext` adds goal and size projections to 128-dimensional features and applies one four-head attention block. `SetArrangement.forward` passes mixed features only as `selector=mixed`; coordinate prediction still receives original `hidden`. The module-level comment is less precise than this actual call, so the diagram must follow the call.
- `partial_arrangement.py::_place_selected` arranges the selected subset, decodes geometry, conditionally repairs infeasible rows, merges held estimates, then searches clearance translations. `arrange_held` returns source estimates if the mover set is empty. No-op is a bypass, not a neural rearrangement followed by reversal.
- Representation widths may be annotated as 12-d objects, 13-d goal, 128-d relation/context. Candidate checking is bounded to at most eight per constrained search; it is not an eight-layer architecture.

## Rendering and evidence requirements

Before and after must reuse the same meshes, material settings, lights, camera matrix, world, framing, resolution, sampling, and color management. Only pose inputs change. The after pose must use saved predictions exactly. Camera elevation and light quality may improve presentation, but the caption must say that the model consumed the original overhead RGB, not the new studio render. Success/failure labels come from existing reports. A 17.657 mm gap or 9.896 mm clearance is best explained with an overhead render crop and a calibrated measurement overlay; a perspective pixel distance is not a world-space measurement.

A front-page successful pair and a later shared-scene full/partial/no-op grid tell complementary stories. Rendered failures are valuable evidence and should not be corrected for aesthetic reasons. A grade cannot be guaranteed; this redesign should make the actual technical contribution easier to judge without expanding its empirical claims.
