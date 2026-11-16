# Figure-set redesign: implementation and evidence audit

Read-only inspection on 2026-11-15, checkout `535f14f`. No model inference, training, checkpoint changes, or changes to figure/manuscript sources were performed for this audit. This note supplements the earlier method audit with direct checks of the currently present source.

## Why the current architecture is inadequate

The `architecture_figure.py` inspected at the start of this revision had reverted to a 214 pt rounded-card diagram. It loses essential connections: the object representation does not connect to relation features, row/order does not connect to the pose heads, object-conditioned referent selection is absent, the goal appears to feed held pose generation directly, and geometric rules sit in disconnected chips below a neural-to-output arrow. The last presentation suggests that rules are optional commentary rather than part of final prediction. Several labels are 5.8–6.6 pt.

The rendering compositions also distort aspect ratio: the motion script maps 1920×1280 images into 192×108 pt, and the architecture maps the output into 96×52 pt. Their corresponding undistorted dimensions would be 192×128 and 96×64 pt. The 340 pt opening figure repeats a full motion board despite a second motion figure later. These are concrete causes of lost visual clarity, independent of color or font choice.

## Exact adopted neural path

The entry point is `experiments/end_to_end/score_end_to_end.py::load_adopted`, followed by `src/knolling_model/experiments/size_refinement.py::poses_from_detection`. The latter receives RGB and English only. The original relation model's displacement and anchor heads are not used by this path; only its hidden encoder is reused.

| Stage | Verified implementation | Appropriate visual representation |
|---|---|---|
| Detector | RT-DETR-R18; native RGB 576×384 resized internally to 640×640; three feature levels, 300 query tokens of width 256, three decoder layers. `rtdetr.py::load_detector` validates these settings. | Three feature levels and a three-layer decoder, rather than a generic “vision encoder” label. Do not show 300 final objects. |
| Physical attributes | Detector center/foreground score, separate physical size, doubled-angle direction, color3 and shape3 heads. `model.py::PhysicalSourceHeads`, `arrangement/perception.py::RecoveryPerception`. | Query token splits into detector geometry and physical attributes. |
| Appearance residual | Predicted-box 16×16 RGB crop; CNN 3→16→32 with pooled32; concat query256 gives288; MLP288→128→10 adjusts size2, angle1, color3, shape3, foreground1. `experiments/architectures.py::AppearancePerception`. | A small crop branch joining the query token. |
| Size residual | Separate 8×8 RGB crop, Conv3→16+GELU+pool; concat query256 gives272; MLP272→64→2; add to base size, clamp≥10⁻⁴m. `size_refinement.py::ObjectSizeRefinement`. | Distinguish from appearance refinement; it is a separate learned stage. |
| Object vector | 12 values: normalized center2, normalized size2, cos/sin(2θ)2, color probabilities3, shape probabilities3. `arrangement/contracts.py`. | A typed object-token strip or matrix, explicitly schematic. |
| Language/global goal | MiniLM tokens384→linear128; seven learned queries attend to text through four-head attention, residual LayerNorm, FFN128→512→128, residual LayerNorm. Keep1 + axis2 + region3 + gap3 + group2 + sort2 produces13 goal values. `arrangement/goal_encoder.py::GoalEncoder`. | Distinguish global goal from object selection. |
| Referent grounding | Objects12→128 feed `JointBlock`: four-head object self-attention, four-head cross-attention to the seven language slots, FFN128→512→128; linear128→1 produces each object's selection probability. | An object-to-language cross-attention branch ending in a separate selection output. |
| Relation representation | MLP12→128→128 with GELU between layers. `relation_pose.py::NeuralRelationPose._hidden`. | The inherited representation h. Do not depict the old relation displacement decoder as adopted. |
| Set context | z=h+Linear13→128(g)+Linear2→128(size); LayerNorm; one four-head attention layer with a learned null key; residual returns h+attention. `set_context.py::ObjectSetContext`. | A single attention block, with an explicit original-h bypass. |
| Row assignment | concat contextual128+goal13; MLP141→128→8. `row_layout.py::NeuralRowPlacement.assign`. | Eight row logits per object, discrete row choice. |
| Within-row pointer | Available-object mean128 + previous token128 + goal13 gives269; MLP269→128→128; scaled dot product with available tokens, autoregressive argmax/masking. | An autoregressive pointer sequence, not an independent slot classifier. |
| Initial row origins | Pool original h per row, plus count/totalwidth/maxheight/firstwidth; own row132 + above rows132 + below rows132 + goal13 + rowcode8 =417; MLP417→128→2. | A row-level coordinate branch fed by discrete row/order. |
| Horizontal step | Previous width, current width, requested gap; MLP3→32→1 and softplus; cumulative X positions within the row. | Geometrically interpretable width-conditioned increments. |
| Initial angle | Original h128+g13; MLP141→128→2, decoded with half-atan2; π-periodic. | Two-vector angle head, not degrees directly. |

**Important precision:** the thirteen-dimensional global goal is produced from text slots only. Object features influence the separate referent/selection branch. It is accurate to call the overall goal encoder object-conditioned, but an arrow suggesting that object features modify global g is not accurate. `SetArrangement.forward` passes original `hidden` into placement and uses `selector=mixed`; contextual features guide row assignment/pointer order, not the coordinate or angle MLPs directly.

The row code also includes a linked-Y correction (`row_anchor`18→1; `row_gap`18→32→1), with zero initialization/fallback for missing older weights. Its presence should not be credited with the v2 release gain without checkpoint evidence. A compact main figure can show row-origin/step/angle heads; exact Y implementation belongs in the appendix if expanded.

## Rules that must remain on the actual output path

- Foreground threshold0.5; duplicate-center suppression within10mm; packed count must be1–8 (evaluated scenes have2–8).
- Literal region/gap consistency correction adjusts the global goal bins. It is not a neural layer.
- Full versus selected-subset versus no-op routing. Unambiguous named-color selection overrides uncertain learned membership.
- Structures are ranked by row/pointer probabilities; bounded checks inspect at most eight candidates per constrained search. Row partition enumeration can examine more states before returning candidates.
- Area order, rotated-envelope geometry, gap-band/region corrections, conditional row repair and color separation determine final movable-object poses. These are material output transformations.
- Held/no-op centers come from detection and angles from RGB silhouette PCA, with learned direction fallback. They bypass rearrangement.
- Partial outputs merge movers and held estimates, then use bounded5mm-grid translations within±120mm for static clearance. This is not trajectory planning.

A main diagram should show distinct visual semantics for learned transformations, discrete/rule-based operations, and data. Neural versus rule legends are preferable to claiming all neural blocks were jointly trained.

## Training boundary verified

Base perception and arrangement were trained separately. The size residual stage updates only that residual. In `experiments/end_to_end/joint_stage.py`, `trainable` returns the horizontal step parameters and final row-origin layer only; the size residual is also unfrozen. Refined size is explicitly detached before object packing; frozen goal/relation computation runs under `no_grad`. Thus there is no pose gradient through the size predictor or perception in the adopted restricted joint stage. Row assignment, pointer and angle heads stay frozen there. The later selector stage trains contextual row/pointer prediction. The rejected broader joint pilot is a separate failed configuration, not an adopted end-to-end training result. The v1/v2 release comparison holds the learned base fixed and evaluates inference-rule changes.

## Quantitative figure opportunity

The strongest message is paired improvement with no lost successes:489 preserved,51 gained,0 lost,100 failed under both releases. Overall strict success is76.40625%→84.375%, difference7.96875pp, paired family-bootstrap95% interval4.84375–11.5625pp. Use this paired structure or gain interval prominently instead of requiring the reader to infer the headline from two small diagnostic panels. Keep the existing task-count table for exact denominators.

The complexity plot is useful secondary evidence: eight-object success47/90→65/90; two-object91/95 for both. The remaining-failure message can avoid overlapping-bar ambiguity: among100 v2 failures,29 have horizontal-gap only,22 vertical-gap only,20 both, and29 neither spacing tag. This partition was directly counted from the frozen v2 JSONL during this audit; it matches the preserved71 unique spacing failures. The “neither” group may contain several other tags; it is not a single failure mechanism. All-error counts can remain in the appendix.

These visual alternatives reorganize existing results. No new inference, rerun evaluation, ablation, or model-selection claim is justified.

## Alternative saved successful illustration

`audit/evidence/full67_choice.json` is a complete six-object scene with original footprints/heights/colors and saved source/predicted target poses. Sample `expanded_e2ee7a75ae3a34826b36d71f_0`: full workspace, horizontal long axes, ascending area,30mm horizontal/vertical gaps, recorded PASS, mean center movement147.55mm. The original RGB path is `artifacts/generalization_20000/dataset/families/0165/source/rgb.png`; README demo `docs/v2.0.0/demo-gallery/motion/11_full_pass.png` uses this example. It is a suitable cleaner opening pair, reserving the existing eight-object sample for detailed correspondence.

The seven-object `full7_choice.json` also contains complete saved geometry/poses (full workspace, horizontal,descending area,20mm,PASS), but its frozen record omits the full instruction string. Prefer the six-object record when reproducible caption wording matters.

Both share the existing qualitative evidence limitation: no per-record checkpoint digest, selected validation illustrations rather than an independent qualitative benchmark. Do not increase “success rate” from these examples. The ten-entry `docs/v2.0.0/demo-gallery/summary.json` is stamped v1 commit26dca53 despite its folder name; its vertical eight-object demo must not be relabeled v2 evidence. The later examples11–13 are linked in HTML but absent from that ten-entry summary.
