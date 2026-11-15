# Dataset contract

The same workspace and asset rules underlie the 100-scene pilot and the sets used for v2. The teachers and the acceptance tolerances do not. Prediction scoring is a third standard, recorded in [evaluation.md](evaluation.md).

## Workspace

Lengths are metres and angles are radians, measured counterclockwise from +x. The table is 0.600 × 0.400 m with its origin at the bottom left. The image is 576 × 384. The conversions are `u = x / 0.6`, `v = y / 0.4`, `px = 576u − 0.5`, and `py = 384(1 − v) − 0.5`. `source_envelope_xyxy_uv` stores world-frame bottom-left bounds, so its vertical axis is not the image axis.

Collision and boundary checks use the oriented major/minor rectangle. The mesh footprint is stored separately and is what the mask projection uses. The rectangle is the conservative envelope: a chamfer or capsule occupies less of the table than the rectangle the metric uses.

Requested regions, in metres `(xmin, ymin, xmax, ymax)`:

| Region | Bounds |
| --- | --- |
| full | (0.02, 0.02, 0.58, 0.38) |
| left | (0.02, 0.02, 0.28, 0.38) |
| right | (0.32, 0.02, 0.58, 0.38) |

Assets are procedural boxes, chamfers, and capsules. The major side is 40–80 mm, the minor side 25–50 mm, and the height 5–20 mm, on a 1 mm grid, with aspect ratio at least 1.3. Colors are red, green, and blue. A scene contains 2–8 objects. Requested gaps are 15, 20, or 30 mm. Rows run left to right and then top to bottom. A row break is a vertical gap above 10 mm. Color grouping requires each row to be a single color and each color to occupy one contiguous run of rows; the gap between different-color bands is twice the requested gap. Size order uses the bounding-rectangle area and allows at most a 5% inversion.

An accepted target keeps at least 10 mm between envelopes and at least 10 mm from the table boundary. The numerical tolerance on that geometry is 0.1 mm, and intersection area above 1e−12 m² is a collision. Locked objects copy the source pose exactly. The movement label is true when the displacement exceeds 3 mm or the major-axis change exceeds 3°. That label threshold is not the 1 mm / 1° keep tolerance used when a prediction is scored.

Object identifiers belong to the source instances and are copied onto the targets. There is no target-side reassignment. Render metadata numbers instances from 1 in source order; annotations keep both identifiers. Target poses, dimensions, and the selected and locked sets are stored as labels. Detection boxes are auxiliary labels. Inference does not read them, and it does not read the target pose.

## Published v2 sets

`artifacts/generalization_20000/dataset` and `artifacts/v2_independent_eval` are described in [dataset.md](dataset.md). Both name the teacher `continuous_rows_v3`. Its stored configuration is axis 2°, gap tolerance 2 mm, and row-center tolerance 2 mm, with continuous envelope translation. `teacher_v3.validate_layout` applies those values. `validator.validate_layout` still defaults to axis 5° and a 5.1 mm gap, which is 5 mm plus the 0.1 mm geometry tolerance. The 100-scene pilot used that default. These two published sets do not.

The 20,000-scene generator is `expanded_data.generate`. It draws each source with `generate_scene(..., mode="partial")`, so the source sampler asks for 12 mm pair clearance and a 12 mm table margin. A full-mode call of the same sampler would use 5 mm pair clearance and the same 12 mm margin. Each family then receives five English instructions from `english_instruction`, under the contract `english_rows_v1`: horizontal full, vertical-axis full, one-color partial, keep-all, and one contrasting full instruction. The asset registry for this generator is `build_registry(120913, 64)`, sixty-four families in each asset split. The scene split is fixed by `partition` before rendering.

The holdout contract records the same registry seed, development assets only, and schema `v2_independent_eval_v1`. The program that built those 128 families is not in this repository. Neither published set is produced by the 100-scene command below.

## Earlier 100-scene pilot

The pilot command is `build_dataset`. It accepts 10 to 100 scenes, in multiples of 10, and the language migration refuses more than 100 source families. The active English artifact of that pilot is `pilot_v1_3_en`, with `language_policy="english_only_v1"` and `instruction_contract="english_rows_v1"`. The schema still parses legacy `ko` text. The English audit rejects it. Parent v1.2 and v1.3 trees are archival. The conversion rewrites instruction text and does not call Blender or recompute a target.

Two teachers exist from that stage.

The legacy teacher groups colors independently, uses one to four objects per row, and searches at most 20,000 candidates. Exhausting the templates or the budget is a generation failure. A disagreement with the independent validator stops generation.

`balanced_rows_v1.3` allows one through N selected objects per row, ranks full arrangements by shape and partial arrangements by translation, and recenters new full blocks. English wording for those labels is in [v1.3-english-conventions.md](v1.3-english-conventions.md). The published v2 sets do not use this policy.

The 100-scene run checks the generator. It is not a performance result.

## Training data usage

The published corpus contains 20,000 scenes, split into 16,000 train, 2,000 validation, and 2,000 test. That total is the generation scale. The official 2,000-scene test and the 640-scene holdout are excluded from training and from checkpoint selection.

The published system is not one training run over those 20,000 scenes. The sequential and relation stages read a 791-scene training manifest and a 94-scene validation manifest drawn from this corpus. That scope is recorded in `configs/neural_sequential_phase1.json` and enforced by `scripts/run_phase1_pairwise_affine.py`. The joint row and size stage, `experiments/end_to_end/joint_stage.py`, reads the train split only. It holds out one fifth of those families, by integer division, for development, and takes its updates from the remaining fit families. It does not open the official test split.

## Package map

The dataset package does not train a network. The model code lives in `knolling_model`.

| Module | Entry | Role |
| --- | --- | --- |
| M0 | `schema.py` | Assets, poses, scenes, instructions |
| M1 | `build_registry`, `make_asset` | Seeded asset pools |
| M2 | `generate_scene` | Source scene |
| M3 | `generate_instruction` | Structured instruction, then English text |
| M4 | `solve` | Target layout and row labels |
| M5 | `validate_source`, `validate_layout` | Independent geometry check |
| M6 | `render_batch`, `validate_render` | Blender RGB and masks |
| M7 | `prepare_annotations`, `build_dataset` | Pilot batch |
| M8 | `audit_records`, `audit_dataset` | Leakage and integrity audit |

M5 never trusts a stored success flag. The audit recomputes family aliases, relation labels, unit conversions, manifest counts, and file hashes. Render checks reproject polygons through pixel centers: per-object IoU at least 0.98, centroid error at most 1 px, interior color agreement at least 99%, and white clipping at most 1%.

Python dependencies for the dataset tools are pinned in `requirements-lock.txt`. Rendering used Blender 5.1.0, Cycles on CPU, 32 samples, and a fixed per-family seed. Sampling uses a local RNG. PNG timestamps are stripped. Another machine or another Blender build is not expected to reproduce the PNG bytes.
