# Architecture figure redesign — 2026-11-15

## Deliverable and reproduction

- Source: `scripts/architecture_figure.py`; callable entrypoint `generate()`.
- Outputs: `figures/architecture_new.pdf`, `.svg`, and `.png`.
- Size: **396 × 225 pt**, matching the manuscript's 5.5-inch text width.
- Smallest text: **7 pt** at the intended insertion size. PDF/SVG retain vector labels, rules, arrows, and glyphs; image thumbnails are embedded rasters.
- Reproduce from the repository root: `paper_submit/.venv/bin/python paper_submit/scripts/architecture_figure.py`.

The exported PNG was inspected directly after two layout corrections. A renderer-based bounds audit checked every visible text artist and found **zero text elements outside the figure**. The initial unsupported Unicode ellipsis was replaced with a rendered mathematical ellipsis; the detector caption was split and left-aligned to prevent clipping. The final figure was reduced from 238 to 225 pt high while retaining 7 pt minimum labels.

## Visual decisions

The figure follows a single horizontal sequence of **representations**, not a stack of colored software boxes: actual source RGB, a bracketed object-feature matrix, a row/order schematic, and an actual predicted-pose Blender view. A separate line shows object-conditioned instruction grounding. A gray dashed route explicitly preserves held/no-op estimates. A thin divider separates the overview from the compact internal feature-routing detail.

The lower detail uses open notation and operation labels. It makes the important original-feature branch visible: contextual features select row and pointer structure; pose heads still read the original relation features. There is no decorative transformer stack, fabricated attention map, or rounded module container.

The primary design references are the data-centric pipeline of **DINOv2**, Figure 3 ([paper](https://arxiv.org/pdf/2304.07193)), and the image-anchored overview/detail hierarchy of **Segment Anything** ([paper](https://arxiv.org/pdf/2304.02643)). The preceding direct visual review is documented in `audit/design_review_2026.md`; the DINOv2 primary PDF was also consulted in this revision. These papers inform visual organization, not additional method or performance claims. No source artwork was copied.

The README demo `docs/v2.0.0/demo-gallery/motion/13_full_pass.png` was opened and inspected. The architecture thumbnail uses the same real scene, but detailed source–target correspondence belongs to the dedicated qualitative figures rather than this compact model overview.

## Data and implementation correspondence

- Source image: `audit/evidence/qualitative_full_source.png`.
- Output thumbnail: `renders/full_after.png`, the saved-prediction render covered by `audit/blender_provenance.json` and `audit/blender_pose_checks.json`.
- The matrix glyph has **12 columns**, matching `arrangement/contracts.py::features`: normalized position (2), normalized size (2), doubled-angle cosine/sine (2), color probabilities (3), and shape probabilities (3). Its group colors identify attribute families; they **do not encode measured values or activations**. Three displayed rows plus an ellipsis denote a variable-size object set.
- The row/pointer circles and letters are **schematic structure symbols**, not recovered object identities, measured attention, or a prediction for the illustrated scene. The caption must retain this distinction.
- `experiments/size_refinement.py::poses_from_detection` supplies detection, refinement, object packing, object-conditioned goal encoding, full/partial/no-op routing, and geometry processing.
- `experiments/relation_pose.py::NeuralRelationPose._hidden` provides the 12-to-128-dimensional relation representation (implemented as 12→128→128).
- `experiments/set_context.py::ObjectSetContext` uses one four-head attention block and goal/size projections. `SetArrangement.forward` sends its mixed features as `selector=mixed`, while passing original hidden features to placement.
- `experiments/row_layout.py::NeuralRowPlacement.forward` uses contextual selector features for row assignment and pointer order, original hidden features for angle, and original pose features plus sizes, goal, and discrete structure for positions. The lower diagram summarizes this routing; the pose label includes both coordinates and angle.
- `experiments/partial_arrangement.py` preserves detected source centers and RGB/PCA orientation estimates for held/no-op objects. The held branch is a bypass, not a neural rearrangement followed by reversal.
- Geometry is an explicit non-learned stage between proposed structure/poses and final output. The diagram does not imply joint end-to-end training, collision-free robot trajectories, or hardware execution.

## Caption guidance

Retain the statement that feature/row glyphs are schematic and that the output image renders saved model predictions. The original source RGB is the model input; the new oblique Blender view is presentation of its predicted poses. Detailed geometry repair, named-color selection, and clearance checks are described in the surrounding manuscript rather than expanded into tiny blocks here.
