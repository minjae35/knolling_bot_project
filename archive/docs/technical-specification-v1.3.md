# Knolling Bot Technical Specification v1.3

Written: 2026-11-03, America/New_York

Reference implementation: `69cf56549efa7f178622e6eadcc6d0eeb6b749d6`

Document status: final integrated specification. The completion record in section 1, the comparison in section 7, and the implementation checklist in section 18 preserve the design baseline at `69cf565`. Dataset v1.3 implementation and migration were finished through `400d9c4`. Those results follow the [v1.3 pilot report](v1.3-pilot-report.md). That report's pending human visual review is kept as the state at that time. This confirmation clarifies re-application of position-based references, legacy orientation compatibility, and the partial evaluation criterion. It does not claim a code change or a newly completed verification for that clarification itself.

Implementation-status update, 2026-11-03: the user confirmed that the final dataset quality review was done and asked for the S0 implementation. The adapter, model, matching and loss, evaluator, and launcher, which keep the existing architecture and learning strategy, and their local checks, are recorded in the [S0 implementation and run contract](s0-implementation.md). The dataset, the teacher, and previously approved sidecars were not changed. Training was not run. A real start comes after this implementation review. Distinguish the design baseline and the old checklist below from what is actually finished.

The goal is a knolling system that predicts a target placement for every object, from a bird's-eye RGB image and a natural-language instruction, with one neural network. Keep the existing M0–M8, the checked annotation contract, and the 100-scene pilot. Change the teacher so that it picks a balanced layout among a limited set of candidates. Initial neural training starts with detection plus target loss. Prefer arrangement success, whether the system can be built, and the ability to name a failure cause over novelty.

This document defines the dataset, teacher, model, loss, evaluation, and simulation contracts together. It separates **facts that are already done**, **changes required in the next implementation**, and **extensions that wait on a condition**. Hyperparameters and performance bars are initial decisions for a run. They are not experimentally confirmed optima or achieved performance.

## 1. Current state and what is preserved

### 1.1 Completed work

| Item | Confirmed state |
|---|---|
| Dataset generator | M0–M8 implemented |
| Tests | the earlier completion report records 39 passed |
| Git | 12 commits; local HEAD and the local `origin/main` record agree at `69cf565`; the main push is taken from the user's report and the earlier report |
| Data | 100 scene families, 130 samples, 210 render views |
| Primary | full 50, partial 30, no-op 20 |
| No-op | 14 repeated instructions, 6 explicit keep-all |
| Counterfactual | 30 families share a source image and use a different instruction and target |
| Acceptance | primary 100/101 = 99.01%; extra instructions 30/30 |
| Automatic audit | 0 errors in geometry, identity, instruction, mask, coordinates, split, and file consistency |
| Human review | a full review of all 130 samples is not done |
| Neural / robot | no evidence that implementation, training, or robot execution is finished; that is the next step |

Sources: the [pilot report](pilot-report.md), the [current data contract](../../docs/dataset-contract.md), and the [implementation decisions](implementation-decisions.md). The 39 tests and the automatic audit are earlier runs. They were not rerun while writing this document. The generation acceptance rate is not a neural success rate or an aesthetic score.

The baseline data to preserve is `artifacts/pilot_v1_2/`. The manifest SHA-256 is `b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27`. In the current provenance, annotation revision `8cda9be` and render revision `47ca812` have a different role from the final-check commit `69cf565`, so they are not overwritten.

### 1.2 Rules for this revision

1. Do not discard existing code or data, and do not regenerate on top of the original.
2. Keep the v1.2 data as a valid legacy-teacher result and as regression evidence.
3. A v1.2 target that already passed the geometry and instruction contract does not become an error just because the v1.3 candidate choice changes.
4. Do not train on a mixture of different target policies for the same input. A training manifest names one teacher policy.
5. The deliverable of this step is the document only. Later code edits, data updates, and training runs are separate implementation work.

## 2. Goal and supported scope

### 2.1 Definition of success

Success means every observed object is matched, and the predicted target layout jointly satisfies collision, minimum gap, boundary, the natural-language instruction, and the keep constraints. The pose does not have to match the teacher coordinates exactly. Target error is an imitation-accuracy metric and is reported separately from task success.

A separate rule-based planner does not edit or decide the final layout. The heuristic teacher is used only to generate training data and as a baseline. A safety validator before execution may only allow or refuse execution. It must not repair coordinates.

### 2.2 v1 supported conditions

| Item | Contract |
|---|---|
| Workspace | 0.600 × 0.400 m, fixed calibrated bird's-eye view |
| Input image | RGB, H=384, W=576 |
| Object count | 2–8; the initial neural pilot starts at 2–4 |
| Shape | box, chamfer, capsule; rigid bodies resting stably on the plane |
| Size | long side a=40–80 mm, short side b=25–50 mm, a/b≥1.3 |
| Height | 5–20 mm; render and physics metadata, not an initial network prediction |
| Dimension samples | 1 mm grid |
| Color | red, green, blue |
| Initial orientation | arbitrary planar orientation, period π |
| New arrangement orientation | long axis horizontal at 0 or vertical at π/2 |
| Instruction | color groups, ascending or descending bounding-rectangle area, gap, full/left/right region, partial edit, no-op |
| Requested gap | 15/20/30 mm |
| Excluded | occlusion and stacking, circles or squares with no distinguishable orientation, visually identical duplicate instances, free-form packing, arbitrary dialogue memory |

Locked objects in a partial request, and an explicit keep-all, keep their existing orientation. Not every target must be 0 or π/2. A vertical-orientation instruction is the direction of each object's long axis. It does not mean that rows should be laid out as columns.

Natural language uses an **English-only restricted grammar** that can name its referent from the current observation. It may refer to a color or to an unambiguous "the leftmost object." Hidden object IDs, instructions that exist only in an earlier turn, and phrases with no interpretation rule such as "a little more" or "like before" are outside the initial scope.

### 2.3 English-only v1 language policy

The only language the initial system supports is English. Every instruction in the active train, validation, development, and test splits must have `language="en"`. User examples, QC, and language-response evaluation also default to English. Korean and multilingual support are optional later work. They are not part of the initial performance target or the training distribution. The language this specification is written in and the language the system accepts are separate.

The 73 Korean and 57 English sentences of the existing `pilot_v1_3` are kept as legacy history. The active data is the 130 English sentences in a separate `artifacts/pilot_v1_3_en/`. English sentences are produced deterministically from the structured instruction fields and the source observation. A free translation or a rerun of the teacher must not change meaning or coordinates. Existing English sentences are also brought onto the same compact contract. A selection set that cannot be expressed as an observable position reference is reported as a conversion failure. It is not replaced with a hidden ID.

Keep, without omission, the long-axis direction, the selected and locked sets, color-pure and consecutive rows, bounding-rectangle area order, the horizontal gap, the vertical gap between row bands, the 2g gap between colors, the region, and the keep meaning. The keep-all example is "Keep every object at its current position and orientation." Do not drop meaning in order to shorten a sentence, and do not arbitrarily turn an already arranged no-op instruction into keep-all.

The final quality review keeps the current restricted grammar for S0. The existing defaults that preserve the meaning of shorter user phrasing are in the [English arrangement conventions](../../docs/english-instruction-defaults.md). The short examples in that note are proposals for a later grammar. They do not change the current annotations or the supported grammar. Do not treat 81 strings that differ only in slot values as 81 kinds of natural English paraphrase. S0 diagnoses overfitting to a fixed grammar. It is not a test of open-language generalization.

## 3. Coordinates and the geometry contract

The origin is the bottom left of the workspace. +x is right, +y is up, and θ is counterclockwise from +x. An object's long axis has period π. The canonical stored range is [-π/2, π/2). π/2 and -π/2 are the same direction.

**Legacy orientation compatibility:** an existing v1.2 source or target may store +π/2. That value is not invalid only because it is outside the canonical range. A newly computed v1.3 orientation is normalized into the canonical range, so vertical is stored as −π/2. An existing source, a locked object, and a no-op target are copied exactly, including their original numeric representation. Existing JSON, images, and hashes are not overwritten for the sake of normalization. Orientation error, `move_label`, geometry comparison, and the migration decision of whether a target changed are computed with period π, so +π/2 and −π/2 are treated as equal. A change of representation alone is not judged to be a move or a target change that needs a re-render. This does not relax the dataset contract that a locked pose is copied exactly.

\[
u=x/0.6,\quad v=y/0.4,\qquad
x=0.6u,\quad y=0.4v.
\]

Pixel centers use the top-left pixel as (0,0).

\[
x_{px}=576u-0.5,\quad y_{px}=384(1-v)-0.5.
\]

`source_envelope_xyxy_uv` is in the workspace frame with the origin at the bottom left. It is not an image y-down box. A mask bounding box is in image coordinates, and its maximum x and y are exclusive.

Ground truth, the teacher, and the validator use meters for length and radians for angle. Only the network's center output is expressed in uv. It is converted back to meters before the physical normalization of the regression loss and before a geometry check. Network size is the intrinsic a and b. It is not the width and height of an image-axis-aligned box.

Projected width and height at the target orientation are

\[
w_i=a_i|\cos\theta_i|+b_i|\sin\theta_i|,\quad
h_i=a_i|\sin\theta_i|+b_i|\cos\theta_i|.
\]

Every safety condition uses the object's rotated bounding-rectangle envelope. The true mesh footprint is stored separately for render checks and physics simulation, and it may not extend outside the envelope. A capsule's smaller true area does not relax a safety condition.

| Check | Value |
|---|---:|
| Full source minimum gap | 5 mm |
| Source minimum boundary margin | 10 mm |
| Source margin of the current partial and keep-all generator | 12 mm gap and boundary, preserved |
| Minimum gap and boundary margin of every valid target | 10 mm each |
| Work region of objects selected by the new teacher | the whole envelope lies inside the inset region below |
| Numerical geometry tolerance | 0.1 mm |
| Overlap area that is not allowed | greater than 1e-12 m² |

Regions are meters in (left, bottom, right, top) order: `full=(.02,.02,.58,.38)`, `left=(.02,.02,.28,.38)`, `right=(.32,.02,.58,.38)`.

## 4. Annotations and version compatibility

### 4.1 Fields that stay

The file formats stay JSON, JSONL, and PNG. **Technical specification v1.3 and `schema_version="1.2"` are different things.** The schema is not bumped to 1.3 in bulk, because the data structure and the meaning of the fields do not change.

| Layer | Existing fields and meaning |
|---|---|
| Sample | `schema_version`, `sample_id`, `scene_family_id`, `split`, `mode`, `role`, `no_op_type`, `status` |
| Instruction | `text`, `language`, `selected_ids`, `locked_ids`, `group_by`, `sort_direction`, `axis_alignment`, `gap_m`, `region`, `keep_all`, `preference_text`, `expected_status`, `reason` |
| Scene | `family_id`, `seed`, `split`, `objects` |
| ObjectState | `object_id`, `asset`, `color`, `source_pose`, `target_pose`, `locked`, `move_label` |
| Asset | ID and family, shape, a/b/h in meters, footprint, mass, friction, provenance, license |
| Relation labels | `rows`, `relations.horizontal_neighbors`, `relations.vertical_row_neighbors` |
| Render and audit | source and target render directory, validation, source_validation, render_validation, file_hashes |
| Derived object labels | instance_id, asset and family, color and shape, dimensions_m, source and target uv and pixel center, envelope and mask box, locked and move |

The derived labels in the last row are not a separate source of truth. They must be recomputable from the original pose and asset. Join on object ID. Do not match source to target by array order alone. Instance ID follows the explicit mapping in the render metadata. 0 is background. 1..N are objects.

`move_label` is true when the center moves by more than 3 mm or the π-periodic orientation changes by more than 3°. It is not the same as `locked`. A selected object can have move=false if it does not need to move.

The model inputs are only the source RGB and `instruction.text`. Selected and locked IDs, structured semantic fields, ground-truth size, pose, mask, and target RGB, the teacher row, and the object ID are for supervision and evaluation. They are not network inputs. `preference_text` is currently an empty reserved field. A preference is written into the explicit instruction sentence.

### 4.2 Metadata to add

Do not insert arbitrary fields into the existing scene or instruction dataclass. Add the following at dataset level in `provenance.json`, and in separate `teacher_audit.jsonl`, `migration.jsonl`, and `human_review.jsonl`.

- Active data: `spec_version="1.3"`, `teacher_policy="balanced_rows_v1.3"`, `instruction_contract="english_rows_v1"`, `language_policy="english_only_v1"`, `supported_languages=["en"]`. The existing `row_edges_v1` is kept as the legacy sentence contract.
- `parent_dataset_manifest_sha256`, the teacher code revision, and the config hash.
- Per sample: old and new target hash, kind of change, template and row size, J_shape, J_move, number of checks, and failure reason.
- Counterfactual pair ID, which semantic fields changed, and the cross-instruction check.
- Reviewer, review time, and a per-sample judgment of meaning, tidiness, naturalness of the edit, and readability.
- Later sequential-edit data records `lineage_root` and `parent_sample_id` as separate metadata.

An existing dataset with no policy metadata is registered explicitly as a legacy policy after its origin is checked. A loader must not treat a legacy target as v1.3 just because it sees the current code version. A checkpoint stores the manifest hash and the teacher policy that were used.

## 5. M0–M8 and where v1.3 changes them

| Module | Input → output | Dependencies | v1.3 work |
|---|---|---|---|
| M0 Schema | asset, pose, instruction → checked structure | standard library | keep existing fields and units |
| M1 Assets | seed and dimensions → registry, mesh, footprint | M0 | keep |
| M2 Scenes | registry, seed, split, N → source scene | M0, M1, geometry | keep existing generation; sequential-edit sources are later scope |
| M3 Instructions | scene and semantic choice → English sentence and semantic labels | M0 | English-only default; legacy grammar is only for explicit regression |
| M4 Teacher | scene + instruction → status, target, rows, relations | M0, placement geometry, M5 | change candidates, the selection score, and the partial-move criterion |
| M5 Validator | source, target, instruction → an independent decision | M0, Shapely | keep the existing geometry and semantic contract; change rules and regression checks |
| M6 Rendering | pose, asset, camera → RGB, mask, metadata | Blender bpy | keep; render only a target that changed |
| M7 Builder | source, instruction, teacher result → annotation and manifest | M0–M6 | fixed-input migration, policy metadata, update only changed targets |
| M8 Audit/QC | dataset → audit, distributions, review material | M0, M5, M6, M7 | add policy mixing, determinism, counterfactuals, and quality comparison |

The existing `build_dataset` requires an empty output directory and at most 100 families. Do not bypass that to generate at scale. Migration targets a new path. It must be a different operation from `prepare_annotations(seed)`, which resamples sources. A teacher change can change acceptance order from the seed, so regenerating everything from the same seed is not the procedure that preserves existing scenes.

## 6. Fixed rules of the heuristic teacher

### 6.1 Shared meaning

The selected set S and the locked set K partition every object with no overlap. Partial row, order, and region conditions apply to S. Gap, collision, and boundary for the whole scene apply to S∪K. Poses in K are copied exactly.

Size order uses the bounding-rectangle area a×b. With color grouping, order inside each color. Without it, order across all selected objects. The teacher sorts on the exact value. Evaluation's 5% inversion tolerance is used only for the accept decision. The teacher fixes color order as red→green→blue. If the instruction does not name an order, evaluation also allows another contiguous group order.

The sort key is fixed as `(sign·a·b, a, b, color, source_y, source_x)`. Shape, asset ID, and input-array order are not tie-breaks. An unobservable correspondence that exists only because of a complete tie is outside the generation scope.

### 6.2 No-op first

If the current scene already satisfies the instruction and the final safety conditions, copy every target pose exactly from the source pose. Then `rows=[]`, `relations={}`, every object is locked, and move is false. Do not move an already correct layout because of the teacher's new balance score.

An explicit keep-all may preserve a safe scattered scene. A repeated arrangement instruction must also satisfy the requested semantic conditions. A scene whose initial gap is at least 5 mm and below 10 mm, and outside the numerical tolerance, is not a valid no-op answer. If keep-all and the final safety conditions cannot hold together, the case is infeasible.

**Re-application of a position-based reference:** a check that reapplies the same structured instruction to the result scene, keeping `selected_ids`, `locked_ids`, and the requested conditions, is an idempotence check on the same referent. Re-interpreting natural language such as "the leftmost object" on a new observation can point at a different object after a move. In that case the selected and locked sets are decided again from the current observation. The previous ID is not kept as hidden dialogue memory. A reinterpretation that changes the selected set is a new semantic input, so it does not unconditionally require no-op. It is no-op only when the current scene already satisfies the newly interpreted instruction and the final safety conditions. If a position reference is ambiguous, for example because of a tie, the existing ambiguous rule applies.

### 6.3 Wider candidates

When the case is not no-op, use k=1..N_s for N_s selected objects. Keeping the per-color order, each group makes the following two partitions.

1. The existing sequential partition: cut into groups of k and leave the last remainder.
2. The balanced partition: for group size n, r=ceil(n/k), q=floor(n/r), s=n mod r. The first s rows have q+1 objects and the rest have q.

For example, n=5 and k=4 keeps both the existing 4+1 and the balanced 3+2. One template applies the same k and the same partition method to every color group. It does not search every per-color combination. Duplicate templates are removed by the index tuple of the selection order. The default candidate set is limited to at most 2N_s.

Partial adds one template that keeps the observed existing row membership. Using the same row test as M5, it reads only the selected objects, and it keeps the existing members of each row only when color purity, contiguity, and size order agree with the new instruction. Dimensions are recomputed for the new orientation and gap. If they do not agree, this candidate is omitted. Do not invent a past structure for a source that has no complete existing row membership.

### 6.4 Candidate coordinates

Each object's w and h are computed at the target orientation. Objects in the same row share a center y, and x satisfies

\[
x_{i+1}=x_i+w_i/2+g+w_{i+1}/2.
\]

Row height is max(h). Row width is Σw+(n_row−1)g. Every row's x center is the common center of the arrangement block. Rows are placed from top to bottom. The gap between row bands of the same color group is g. The gap between different color groups is 2g.

A row band is the horizontal band that contains every envelope in that row. This gap does not mean that every pair of individual objects in different rows is g apart. If objects of different heights share a row, the shorter objects have a larger clearance. Keep this meaning in the natural language and in QC.

### 6.5 Choosing a full arrangement

Place each template's block center at the center of the requested region, and compare candidates that pass the full geometry and semantic checks. Let the usable region width and height be R_w and R_h, and the block width and height be B_w and B_h:

\[
J_{shape}=\max(B_w/R_w,\ B_h/R_h).
\]

The order is ascending `(J_shape, B_w·B_h, row count, fixed template key)`. Float comparison uses a key that rounds J_shape and area to 12 decimal places. The last key is the template key defined by the observation-based object order. Raw values are kept in the audit report.

This criterion is an engineering choice that reduces excessive horizontal or vertical extent relative to the region. It does not guarantee an aesthetic optimum. If the 100-scene comparison is regularly worse, review only this score before a large generation run.

### 6.6 Choosing a partial edit

For each template, use the existing global 5 mm translation lattice on which the whole block lies inside the requested region. Locked objects do not move. The cost of a candidate is

\[
J_{move}=\frac1{|S|}\sum_{i\in S}\|p_i^{target}-p_i^{source}\|_2^2.
\]

Sort every template and translation pair by `(J_move, J_shape, block area, row count, template key, -center_y, center_x)` and take the first valid candidate. J_move is in m². The comparison key rounds to 12 decimal places. The initial implementation spends the budget in this global order. An implementation that searches each template separately and then exhausts the budget can change the result, so it is not used.

The old rule, which returned the first valid placement of the first template in order of proximity to the region center, is removed. The existing templates and the 5 mm lattice remain, so an existing valid partial candidate is still in the candidate set. Minimum movement means minimum inside this finite set. It does not mean a global continuous optimum.

The geometry-check budget stays at the existing initial value of 20,000. Cost sorting is not counted in the number of geometry checks. If no valid result is found inside the budget, the status is `generation_failure/search_budget`. If every candidate fails, the reason is `candidate_family_exhausted`. That is not evidence of infeasible.

### 6.7 Consistent results

The same observed scene and an instruction with the same meaning return the same per-ID target. Language, paraphrase, lighting, background, array order, and asset ID are not used as the random seed of the placement policy. The answer can change if object geometry, the current pose, the selected set, or a requested condition changes.

Reapplying a structured instruction with the same selected set, locked set, and requested conditions to the teacher result must be no-op. Re-grounding natural language on a new observation follows section 6.2. Do not mistake a change of referent in a position-based reference for a determinism failure or a no-op failure. A reapplication check that keeps the old IDs does not also verify the ability to reinterpret a position reference on a new image. A row membership that changes when a dimension changes by 1 mm is possible at a threshold. Record such a switch for tagging and visual review. Do not add a random answer for the same input.

## 7. Comparison with the current teacher

The baseline is `69cf565`: [teacher.py](../../src/knolling_dataset/teacher.py), [instructions.py](../../src/knolling_dataset/instructions.py), [builder.py](../../src/knolling_dataset/builder.py), and [validator.py](../../src/knolling_dataset/validator.py).

| Item | Met now | v1.3 action |
|---|---|---|
| Swap a and b after the target rotation | met | keep; compare the general formula with 0° and 90° fixtures |
| Object-edge gap and row max-height | met | keep |
| Independent rows per color and contiguous groups | met | keep |
| Bounding-rectangle area and within-group order stated | met | keep |
| No-op exact copy and empty relation labels | met | keep |
| Source–target IDs and locked poses preserved | met | keep |
| Candidates k=1..4 and a single chunk | not met | k=1..N_s, existing plus balanced partition |
| Score that prefers fewer rows | not met | J_shape first |
| Partial prefers the region center | not met | J_move first, plus the existing-row candidate |
| Determinism for the same seed | the basic implementation meets it | stronger invariance checks for array order, language, and reapplication |
| Row-gap sentence | the number agrees; the row-band wording is unclear | state the row-band meaning in the active English sentence; keep the number and the semantic fields |
| Counterfactual | 30 families exist, mostly an axis change; region can also change to full | record which semantic fields changed, check the crossed conditions, and obtain single-factor pairs in a later extension |
| Failure distinguished from infeasible | met | keep |
| Independent geometry validator | met | keep; do not combine it with the teacher score |
| Full human review of layout quality | not done | review and record all 130 samples |

An existing counterfactual is not an error only because it changed two meanings at once. Separating the language understanding of axis, gap, and region does require single-factor pairs in a later extension.

## 8. Preserving the existing pilot, and migration

### 8.1 Procedure

1. Freeze the manifest, provenance, and audit of the existing `pilot_v1_2` as the baseline record. Do not modify the original files or the existing review state.
2. Read source, asset, instruction semantic fields, IDs, and split from the 130 existing annotations. Do not resample sources.
3. Apply the v1.3 teacher to the same input and compare targets by ID. No-op is decided by whether the existing source already satisfies the meaning, so a legacy arrangement shape can be preserved.
4. Record, separately, samples with no change, samples whose target changed, samples whose sentence was only clarified, and generation failures.
5. Copy the existing source RGB, mask, and metadata into the new dataset path after a hash check. Do not use a mutable hard link that dangerously shares the original.
6. If the target is the same, reuse the existing target render. If it differs, render only that target. No-op reuses the source render.
7. For a changed sample, update together the target pose, derived uv, pixel, and mask box, rows, relations, locked and move, validation, file hashes, the annotation hash, the manifest, the split index, and QC.
8. After a full audit of all 130 and a human review, write a new completion record. Do not hide a failed item and mark the v1.3 dataset complete.

The 100/101 in `generation.json` is kept as the legacy original generation history. The pass rate of the fixed-input migration of 130 samples is recorded separately. Applying a v1.3 target does not relabel the old acceptance rate as the new teacher's generation performance.

### 8.2 What has to be rebuilt

| Change | Work required | Work not required |
|---|---|---|
| Teacher result is the same | re-check and record under the new policy | re-render source or target |
| Target pose changed | update the target annotation, render, derived labels, and audit | regenerate source, asset, or split |
| Sentence clarification only | update text, annotation hash, manifest, and QC, and confirm the meaning is the same | re-render images |
| A new instruction with new meaning | a new sample and a new target on the same source | overwrite the existing sample |
| A new follow-up observed scene | a new source ID and render, and a lineage record | replace the source of an existing family |

Current M8 allows only one source physical scene per `scene_family_id`. A follow-up observation gets a new scene-family ID and inherits the original `lineage_root` and split. Replacing the source of the original family raises `family_source_drift`. The episode split check extends to `lineage_root`. This extension is work for the interactive stage, not a required part of the initial teacher migration.

### 8.3 Legacy compatibility checks

The current regeneration check in `tests/test_pilot.py` calls the current `prepare_annotations` and compares it exactly with v1.2. If the default teacher changes, this check can mistake an intentionally different target for a regression error.

The follow-up implementation keeps the old rules explicitly as the `legacy_v1_2` policy, and the existing reproduction check calls that policy. v1.3 has its own policy reproduction check. The independent geometry and hash audit of the legacy dataset must keep passing. Do not delete an existing test, and do not overwrite an expected value with the new target in a way that erases the original history.

The training loader requires a single active teacher policy. A legacy sample whose target was unchanged in the v1.3 re-evaluation may be reused in the v1.3 manifest. Do not mix competing targets from different policies into one training set. A training run that compares against legacy is a separate run.

### 8.4 English-only sentence migration

Do not modify `pilot_v1_3`. Create `pilot_v1_3_en` in a new directory. In an active annotation, change only `instruction.text` and `instruction.language`. Preserve exactly the other semantic fields, source and target poses, object IDs, assets, and sizes, relations, move and lock labels, scene family, and split. Copy source and target RGB, masks, and render metadata as independent files with the same hash. Do not call Blender. A QC overlay is derived review material. It does not change the model-input image.

Rewrite the annotation hash, the full and split manifests, and QC. `language_migration.jsonl` records, per sample, the previous language, the text hash, the semantic hash, and the English length. `language_migration_summary.json` reports the active language count, semantic mismatches, and zero re-renders. Keep the original generation and geometry-migration statistics as history. Do not confuse them with the current language distribution. `language_parent_manifest.jsonl`, `language_parent_annotations/`, and the existing `parent_annotations/` are for origin checks and are excluded from training. The loader uses only the active `manifest.jsonl` and its split index. It does not glob every annotation in the directory.

After the language change, the human-review state stays pending and `training_approved.jsonl` is empty. The new generation default in the code and the CLI is the balanced teacher plus English-only. The default audit also requires the English contract. A legacy regression audit is separated as an explicit `--legacy --read-only`. It cannot be used to turn off the non-English check of the active English data.

## 9. Dataset generation and quality checks

### 9.1 Pipeline

The order is: fix the asset and geometry-family split, generate or read the source, choose the semantic conditions, write the English sentence, run the teacher, run the independent validator, render with Blender, check RGB, mask, and projection, write the annotation and manifest, audit the whole set, then human review. When only the language of existing material is converted, skip teacher placement and rendering, following section 8.4.

Keep the direct Blender bpy backend. The existing environment is Blender 5.1.0 and host Python 3.12.16. RGB is Cycles on CPU with 32 samples. The instance mask is the Workbench flat object color of the same mesh and camera, converted to uint16 IDs after anti-aliasing is disabled. Keep the normalization that strips PNG time and render-time metadata. A bitwise-reproduction claim is limited to the same Blender build, device, and platform. Do not bring BlenderProc back.

Camera calibration, resolution, and output orientation stay on the existing render-job contract. Initial neural training does not use arbitrary crop, rotation, or left-right flip augmentation. Those transforms would also have to change coordinates and a "left/right" instruction. Hue augmentation that changes a color name is not used either.

### 9.2 Automatic criteria to keep

- Every active split has `language="en"`, and the text matches the approved English grammar built from the source and the structured fields. Korean whose language field was merely set to en, another language written in ASCII, and a wrong number, referent, or direction are rejected. This is a consistency check of the restricted grammar. It does not claim a general language detector.
- Fields other than text and language, and the source and target file hashes, match the original. Each split index matches exactly the corresponding subset of the new manifest.
- Object ID, count, asset, and color are preserved. Locked poses are copied exactly. Units, coordinates, and relations are recomputed.
- Target gap ≥10 mm, boundary ≥10 mm, no envelope collision, and inclusion in the instructed region.
- Mask-to-true-footprint IoU ≥0.98, and center projection error ≤1 pixel.
- Interior RGB color agreement ≥99%, and clipped white pixels ≤1%.
- Source and target instance-ID mappings agree. Training RGB must not contain QC IDs or arrows.
- Zero split overlap of scene, asset, normalized shape, or the same physical scene.
- Target consistency under a change of source or instruction array order, under a sentence with the same meaning, and under reapplication of a structured instruction that keeps the same selected set, locked set, and requested conditions. A check that reinterprets a position reference on a new observation is separate (section 6.2).
- The validator rejects fixtures with a bad ID, a rotated dimension, a locked pose, a unit error, or a 5–10 mm no-op.

### 9.3 Diversity and language checks

The active language distribution is 100% English. Evaluate English instruction types, observable references, changes of sentence meaning, and counterfactuals. Do not count a legacy Korean regression check as an active multilingual evaluation or as a training sample.

Aggregate N=2..8, the three shapes and colors, aspect-ratio and size combinations, horizontal and vertical, 15/20/30 mm, and the full, left, and right regions by condition. Do not let color be fixedly tied to size or shape. Include one color, two colors, three colors, and unbalanced compositions such as 5+2+1. Do not force a particular row frequency to be uniform. Compare the distribution inside N, color composition, and region.

On a counterfactual pair, evaluate each target against the other side's instruction, not only the difference of semantic fields. A diagnostic pair must have each target violate at least one condition of the opposite instruction. If an existing pair fails this bar, the individual sample's answer is kept and the pair is excluded from the count of valid language-diagnostic pairs. Add only the missing pairs that are needed.

The current 100 scenes are a 70/10/10/10 family split and an 89/13/13/15 sample split. This is a small geometry-OOD consistency pilot. The small sample count does not demonstrate a neural success rate above 95%. Validation selects checkpoints. Development diagnoses failures. Test is the final evaluation of a frozen policy. The existing 100-scene test was already exposed during data QA, so a later claim of final performance is evaluated on a newly locked test set.

### 9.4 Order of data expansion

The initial 100 scenes are for format and quality checks and for a small overfit inside the train split. The planned size of a learning-generalization pilot is train 2,000 / validation 400 / development 1,000 / locked test 1,000 samples, starting at N=2–4. Every English wording, image, and command derivative of the same family stays in the same split. Multilingual derivatives are not generated now. An episode whose source changed also keeps the split at the lineage unit.

The initial pilot keeps the basic full, partial, and no-op meanings, and no-op training. A curriculum that starts from one basic arrangement instruction is allowed. Keep the geometry-OOD split. Do not move another split's assets into train in order to raise performance.

The later target of 50,000 positives (full 25,000, partial 15,000, no-op 10,000) and 5,000 negatives for status is an expansion budget. It is not an obligation to generate them now. Expand after the initial neural pilot and the storage, throughput, and chunk/resume checks pass. The negative plan is ambiguous 2,000 / infeasible 1,500 / unsupported 1,500. Do not label a search failure as infeasible.

Keep the current 30 scattered partial sources. Edits of an already arranged state, and 3-turn evaluation, are supplied as separate episodes in a later interactive stage. Do not report that the current 100 scenes have verified that capability.

## 10. End-to-end neural architecture

User-approved architecture extension (2026-11-03): add a separate S0 backend that connects a pretrained RT-DETR-R18 as the internal neural perception. The ResNet-18 defaults below are kept as the contract of the earlier experiment. Differences in RT-DETR query count, preprocessing, gradient path, and intermediate detection supervision are stated in the [integration contract](rtdetr-integration.md). The dataset, teacher, ground truth, and evaluation gates do not change. An internal trainable detector is allowed. An external rule-based layout planner is not added.

### 10.1 How the pieces connect

Source RGB → image backbone and FPN → object decoder → source prediction and object token of the same query → three-stage arrangement decoder → target pose of the same query.

Natural language → token encoder and projection → text cross-attention in the arrangement decoder.

The status head is a reserved separate output. It is not used for training or for a decision at the start. Do not put an external detector, a semantic parser, the teacher, a packing optimizer, or the ground-truth object count on the inference path.

### 10.2 Implementation defaults

| Piece | v1.3 contract |
|---|---|
| Image input | B×3×384×576, RGB/255, then the backbone's pretrained mean and standard deviation |
| Backbone | ImageNet-pretrained ResNet-18, batch-norm statistics frozen at the start |
| Image features | stride 8/16/32 FPN, each d=256; 2D position and level embeddings |
| Object queries | Q=12, d=256 |
| Object decoder | 4 transformer blocks; query self-attention, image cross-attention, FFN |
| Shared transformer | 8 heads, FFN=1024, GELU, dropout=0.1, pre-layer-norm |
| Text encoder | token features of paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions, then a linear map 384→256 |
| Arrangement decoder | 3 blocks; query self-attention, text cross-attention, FFN; uses the object token and a per-stage geometry embedding |
| Source heads | objectness, center uv, intrinsic a and b, π-periodic orientation, color 3-class, shape 3-class |
| Target heads | per-stage center uv and orientation residual |
| Selected output | objectness ≥0.5; no NMS and no ground-truth top-k |

The backbone, FPN, object decoder, and arrangement decoder are one graph. Freezing the text encoder at the start is a training policy that uses a pretrained neural representation. It is not an external rule-based language processor.

Applying the English-only policy does not change the neural architecture above or the current text-encoder choice. The word "multilingual" in the encoder name does not mean the initial system supports multiple languages. The benefit of an English-only pretrained encoder can be compared later in a neural pilot. It is not part of this dataset change.

Even when an English instruction is shortened, **measure the real token length and forbid silent truncation**. The maximum length is 512 including special tokens. Dynamic padding inside the batch stays the default. An input that exceeds the model config's position limit of 512 is reported as an error. Do not treat a word count as a token count, and do not quietly drop meaning. This is an input-preservation setting, not a change to the core network. [Official model config](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/raw/main/config.json)

### 10.3 Head representation and refinement

The source center outputs uv through a sigmoid. a and b are positive and a≥b, parameterized as `b=softplus(z_b)+ε` and `a=b+softplus(z_delta)`. ε=1e-4 m. The initial bias is set so that the output is approximately a=0.060 m and b=0.035 m. A prediction outside the ground-truth range is corrected by the loss. It is not replaced by the ground-truth size at inference.

The source orientation head outputs two real numbers (c, s), read as θ=0.5·atan2(s, c). The initial bias is (1, 0). Epsilon stabilization is applied near length 0. The loss is the π-periodic angle error. A separate orientation-normalization loss is not added.

The initial target pose is the predicted source pose. The geometry embedding at stage k uses the predicted source uv, a, b, orientation, and objectness, and the previous target uv and orientation. Orientation is represented as cos(2θ), sin(2θ). Size is represented as a/0.6 and b/0.4. These are not replaced by ground-truth values or by a ground-truth selected or locked mask. The embedding is projected to 256 dimensions by an MLP and added to that query token.

\[
\hat u^k=\sigma(\operatorname{logit}(\operatorname{clamp}(\hat u^{k-1},10^{-4},1-10^{-4}))+\alpha_k\Delta u^k),
\quad \alpha=(1,0.5,0.25).
\]

v is the same. Orientation is updated as θ^k=wrapπ(θ^(k−1)+(π/2)tanh(Δθ^k)). The last linear layer of the residual head is initialized to 0. The initial output copies the source, but every stage learns to move under target supervision. This coordinate step scale and the stage weight of the loss are different values.

Ground-truth matching is an assignment outside the gradient path. The path from the source prediction through the geometry embedding to the target, and the refinement between stages, is not detached. At the start all 12 queries take part in attention. Predicted objectness is not used to hard-mask ground-truth objects during training. Output cleanup uses only the final objectness threshold.

A separate keep/move decision that forces the target back to the source is not used. `move_label` stays as a diagnostic. Whether an object needs to move can be judged from the difference of a valid source prediction and a valid target prediction.

### 10.4 Neural module interface

| Module | Input | Output | Dependencies and checks |
|---|---|---|---|
| N0 Dataset adapter | fixed manifest, annotation, source RGB | image tensor, token IDs, text mask; a separate supervision bundle | M0, M7, M8, tokenizer; confirm ground truth is not mixed into the input |
| N1 Visual and object model | image tensor | B×12×256 object tokens, source-head tensors | backbone, FPN, object decoder; check source-geometry units |
| N2 Arrangement model | object tokens, predicted source, text tokens | B×12×(uv, θ) at 3 stages | keep the same query index; do not use ground-truth size or lock |
| N3 Criterion | source and target predictions, ground-truth supervision | one source matching, a scalar total, per-term statistics | section 11; valid-sample mask and gradient checks |
| N4 Trainer | N0, N1, N2, N3, run config | checkpoint and train and validation records | section 12; data and policy hashes fixed |
| N5 Evaluator | raw prediction, evaluation ground truth | count, error, geometry, SSR, failures by condition | section 14; distinct from the dataset validator's exact conditions |
| N6 Execution adapter | predicted source and target, calibration | allow or refuse, execution result, re-observed RGB | section 15; do not edit target coordinates |

The object key in an inference result is that image's `query_index`. Each entry includes objectness, source center, size, and angle, color and shape probabilities, and the final target center and angle. Do not call this the ground-truth object ID. A real grasp uses the predicted source position. The match to a ground-truth ID is made only in evaluation. Do not assume that a query number is stable on a newly observed image.

## 11. Initial loss and identity matching

### 11.1 Normalization

Define Smooth L1 as ρ(z)=z²/2 for |z|<1, and |z|−1/2 otherwise. Divide the error, restored to meters, by s_p=.010 m, s_s=.005 m, and s_θ=π/18 rad. These are loss scales. They are not the success tolerance.

\[
\ell_p(\hat p,p)=\tfrac12\sum_{d=x,y}\rho((\hat p_d-p_d)/s_p),\qquad
\ell_s(\hat s,s)=\tfrac12\sum_{d=a,b}\rho((\hat s_d-s_d)/s_s).
\]

\[
\Delta_\pi(\hat\theta,\theta)=\tfrac12\operatorname{atan2}(\sin(2(\hat\theta-\theta)),\cos(2(\hat\theta-\theta))),\qquad
\ell_\theta=\rho(\Delta_\pi/s_\theta).
\]

The error between 179° and 1° is 2°. An exact 90° difference is a non-smooth boundary of the periodic distance, so the finite-value and gradient fixtures include it. [SmoothL1 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

### 11.2 Source matching

Cost between ground-truth object i and the last object-decoder query j:

\[
C_{ji}=-q_j+\frac{\|\hat p_j^s-p_i^s\|_1}{2s_p}
+\frac{\|\hat s_j-s_i\|_1}{2s_s}
+\frac{|\Delta_\pi(\hat\theta_j^s,\theta_i^s)|}{s_\theta}.
\]

Hungarian matching assigns N ground-truth objects to N distinct queries. This π of one forward pass is reused for detection and for all three target stages. The next forward pass computes it again. Do not rematch using target coordinates, the teacher row, or the target order. A query number does not have to mean the same object across scenes.

Color and shape are left out of the initial matching cost and are supervised after the assignment. A separate identity contrastive loss, GIoU, and query denoising are not required at the start. The reason for one-to-one set matching is [DETR](https://arxiv.org/abs/2005.12872). The physical-unit cost above is this project's fixed initial setting.

### 11.3 Detection

For the matched query set M, objectness q, and negative weight η=.1:

\[
L_{obj}=\frac{-\sum_{j\in M}\log q_j-\eta\sum_{j\notin M}\log(1-q_j)}{N+\eta(Q-N)}.
\]

The real implementation uses logit-based binary cross-entropy.

\[
L_D=L_{obj}+\frac1N\sum_i[\ell_p(\hat p_{\pi(i)}^s,p_i^s)+\ell_s(\hat s_{\pi(i)},s_i)
+\ell_\theta(\hat\theta_{\pi(i)}^s,\theta_i^s)+.1CE_{color,i}+.1CE_{shape,i}].
\]

Initial detection supervision is applied to the last object-decoder output. Intermediate object-decoder supervision is added only if S0 shows a convergence problem of the last output, as a supplement that averages the same L_D under the same final matching. There is no independent per-layer weight search.

### 11.4 Target and the whole initial objective

\[
L_T^{(k)}=\frac1N\sum_i[\ell_p(\hat p_{\pi(i)}^{t,k},p_i^t)+\ell_\theta(\hat\theta_{\pi(i)}^{t,k},\theta_i^t)],
\]

\[
L_T=.25L_T^{(1)}+.25L_T^{(2)}+.50L_T^{(3)},\qquad \boxed{L_{initial}=L_D+L_T}.
\]

Every ground-truth object is included in the target loss. Matched target supervision is not removed because objectness is low. Locked targets of partial and no-op are included as they are. Per-stage supervision is auxiliary supervision of the same objective. It is not three separate objectives. The precedent for intermediate decoder supervision is the [official DETR implementation](https://github.com/facebookresearch/detr/blob/main/models/detr.py).

Average per scene, then average over the batch. The loss total must not change automatically because N, the query count, or the relation count differs. The default batch sampler picks a family uniformly and then picks an available instruction sample of that family uniformly. A family with many counterfactuals is not over-weighted. One epoch is defined as one pass over a random permutation of the train families. Instructions inside a family are chosen from a shuffled cyclic list so that all of them are used across epochs. The last small batch is used.

| Item | Initial weight |
|---|---:|
| Detection / target | 1 / 1 |
| Detection position / size / orientation | 1 each |
| Color / shape cross-entropy | 0.1 each |
| Unmatched objectness | 0.1 |
| Target stages 1/2/3 | 0.25 / 0.25 / 0.50 |
| Collision / boundary / relation / keep / status | all 0 |

## 12. Training stages and default settings

### 12.1 Optimizer and run records

Use AdamW, backbone learning rate 1e-5, decoder, head, and projection learning rate 1e-4, weight decay 1e-4, effective batch 16, and global gradient-norm clip 1.0. The text encoder is frozen at the start and the projection is trained. A frozen encoder is in eval mode so dropout is off. Weight decay is not applied to bias or layer-norm. S0 uses FP32, no augmentation, and dropout 0 in the trained modules, to check pure overfitting. From S1, the standard dropout 0.1 is applied. AMP is chosen later, after a numerical check.

After one epoch of learning-rate warmup at the start of a stage, cosine decay lowers the learning rate to 0.1 times the starting rate by the end of that stage's budget. S0 uses a fixed learning rate. Record validation every epoch and a development diagnosis every 5 epochs. A checkpoint stores the model, optimizer, scheduler, RNG, token settings, data manifest and teacher policy, loss scale, and seed.

Checkpoint selection is overall task success, then raw geometry on a tie, then target-center error on a further tie. In S1, source-center error is used after count accuracy. Do not select on a result that a separate planner repaired from the raw prediction. Do not use the test set to adjust weights.

### 12.2 Order and gates

| Stage | Purpose / data | Loss / budget | Pass criterion |
|---|---|---|---|
| S0 | overfit 32 reviewed train samples, including full, partial, and no-op | L_D+L_T, at most 2,000 updates | count 100%; source and target center P95≤1 mm; size P95≤1 mm; orientation P95≤2° |
| S1 | prepare N=2–4 detection; planned train 2,000 | L_D, at most 20 epochs | validation count ≥95%; source center P95≤5 mm; size P95≤3 mm; orientation P95≤5° |
| S2 | joint end-to-end training, starting from simple instructions | 5 epochs with λ_T from 0 to 1, then L_D+L_T; first budget 100 epochs total | raw geometry ≥90%, SSR ≥80%; after the first seed passes, confirm 3 seeds in total |
| S3 | extend to N=2–8 and the instruction range | the confirmed minimum loss; continue training the existing checkpoint | count ≥99%, geometry ≥97%, SSR ≥90%, per-instruction SSR ≥85% |
| S4 | partial, no-op, and follow-up edits | the base loss; keep if needed | partial success ≥90%, no-op ≥98%, scenes with a locked-object violation ≤2% |
| S5 | final evaluation on the locked test | evaluate after training | count ≥99%, geometry ≥99%, SSR ≥95%, per instruction ≥90%, geometry-OOD ≥85%, 3-turn overall success ≥85% |

The S0 numbers diagnose memorization. They are not generalization performance. Even in S0, record the number of safety and semantic errors separately and inspect the remaining failures. Do not hide an S0 failure with an auxiliary geometry loss. Fix units, matching, target ID, mask, and the gradient connection first. After S0, formal training starts as a new run from the same pretrained initialization.

When the predicted source is ready in S1, go to S2. If the budget is spent and the gate still fails, classify the failure instead of only adding epochs. Do not apply a larger N and a new instruction at the same time. Extend one axis at a time. The selection and the sampler of existing data may change. Do not edit the teacher or the ground truth to fit the model.

The current split is already geometry-OOD. Do not pass by applying SSR ≥95% and OOD ≥85% separately on the same test set and then accepting 85%. Use the 85% bar separately only when a harder in-range stress set was defined in advance. The 3-turn bar is evaluated only after episode data and execution re-observation exist. It is not a prior gate of the single-scene neural pilot.

After S2, keep the source-to-target gradient. If joint training makes detection worse, the first limited adjustment is λ_T=0.5. Do not introduce a large weight grid search or automatic multi-task weighting.

### 12.3 Failure diagnosis and the smallest alternative

| Observed failure | Check first | Next action that is allowed |
|---|---|---|
| Loss does not fall in S0 | meters versus uv, angle period, source matching, target ID, gradient, output range | fix the error and re-check the same 32 samples; do not add a new loss type |
| Missing or duplicate sources | objectness target, unmatched average, 12 queries, input mask | average the same detection supervision onto the existing object decoder's intermediate outputs |
| Source is accurate and the target does not move | gradient after residual initialization, detach between stages, text token length | fix the connection or the mask; a temporary ground-truth input is only a separate oracle diagnostic and is not put on the training path |
| The same layout when the command changes | text truncation, cross-attention mask, counterfactual answer difference | fix the input error; if it persists, one alternative is to unfreeze the last 2 blocks of the text encoder at learning rate 1e-5 |
| Source gets worse under joint training | target gradient and learning rate, source metrics | a limited comparison at λ_T=0.5 |
| Source is fine and only gap or boundary fails | error size, margin by condition, final footprint | add only the one term in section 13 |
| Train succeeds and validation fails | split and shape range, overfitting, image and sentence distribution | check coverage and allowed augmentation; do not tune on the test set |

If S2's gate is still not reached on 3 seeds after these actions, collect the failure cases and the learning curves and return to an architecture review, instead of stacking more losses. Another model, or a relaxation of the end-to-end condition, is not included in v1.3 automatically.

## 13. Conditional introduction of auxiliary losses

### 13.1 Shared rules

Review a term when the same failure repeats on three consecutive validations. Separate the cause on scenes where detection is correct, but the overall SSR still counts a miss or a false detection as a failure. If the conditional diagnostic sample is smaller than 100 scenes, do not fix a rate threshold from a few cases of the small 100-scene data. Review the case count and the cause first.

Add one term at a time. Compare against a control run that continues the same checkpoint for the same number of epochs without the auxiliary term. The added weight goes from 0 to 1 over 5 epochs and then stays at 1. That value assumes the normalization and reduction defined below. Do not copy a weight onto a different raw loss. If the effect is too large, compare only the limited alternative 0.3. If there is no improvement, remove it.

| Term | Condition for considering it | Cause to rule out first |
|---|---|---|
| Collision | a 10 mm gap violation persists on more than 2% of scenes where source recognition is correct | a miss, a size error, or an identity error |
| Boundary | a boundary or requested-region violation persists on more than 1% | a coordinate conversion, or a check that looks only at the center |
| Relation | target position improved, but a particular gap or row relation still fails on more than 10% | text truncation, an ignored command, or a bad condition label |
| Keep | partial locked-object violations on more than 2% of scenes, or no-op below 98% | a ground-truth copy, a lock, or an angle-period error |
| Status | the stage that learns to refuse a negative instruction | missing negative data, or a wrong infeasible label |

Status is not a term that patches a geometry failure. An initial pilot that evaluates only valid requests does not allow or refuse execution from an untrained status output. Claiming a final refusal capability requires separate training and evaluation.

### 13.2 Geometry loss

Apply it first to the final target stage. Build rectangles from ground-truth a and b and the predicted θ and p, and include every ground-truth object. Do not allow the shortcut that computes collision from the predicted size and shrinks the object. This use of ground truth is supervision only. The inference path still uses the predicted size.

Collision is L_C=mean_pairs ρ(max(0, .010−d_signed)/s_p). d_signed is the true Euclidean shortest distance between rectangle polygons when they do not overlap, and the negative of the minimum separating depth from SAT when they do. The external shortest distance is a differentiable tensor operation, the minimum of corner-to-segment distances. Overlap depth is computed on the rectangle's four unit axes. Do not use an SAT projection gap as the non-overlap shortest distance. Include axis-aligned, contact, containment, and diagonal-separation fixtures.

The separation depth on axis n when the rectangles overlap is r_i(n)+r_j(n)−|(p_i−p_j)·n|, and the minimum over the four axes is used. r is the radius of the rectangle projected onto that axis. A simple interval-overlap length differs from this value when one rectangle contains the other, so it is not used.

Boundary computes, for the rectangle's xmin, ymin, xmax, and ymax, the shortfall in each of the four directions past the table inset of 10 mm, and averages ρ(shortfall/s_p) over directions and objects. For a selected object, the shortfall against the four boundaries of the requested region is computed the same way and added. The two parts are each averaged over objects and then summed. Do not force the requested-region condition onto a locked object.

The loss does not prove real execution safety between objects. Before it is turned on, confirm that L_C and L_B are zero within numerical error on every corresponding ground-truth case. Target supervision remains even when the geometry gradient is symmetric, as in a complete overlap.

### 13.3 Relation and keep

Relation applies mean ρ((prediction−ground truth)/s_p) to the horizontal-neighbor edge gap and the adjacent row-band gap in the existing `relations`. Add the y difference inside a ground-truth row, on the same scale, only when row-center alignment has to be repaired. The active relation kinds are fixed in the run config, and the per-kind means are averaged again. A no-op, and a sample with no relation, contributes 0. Batch normalization averages only the samples for which that term is valid. Do not re-align an already tolerated no-op to an exact gap.

Keep averages ℓ_p+ℓ_θ between the final target and the ground-truth source on the locked set K. If K is empty, the term is 0. This term increases the weight of locked objects inside target supervision. It is not new answer information. Do not hide a case where both poses are wrong by comparing against the predicted source.

### 13.4 Status

Use the existing 4-class head (ok, ambiguous, infeasible, unsupported). Mean-pool the tokens, concatenate the object and text representations, and output them through an MLP. At the start only the head is reserved, and its weight is 0. The existing rendered pilot is entirely positive, so it cannot be evidence that status training is done.

When an approved negative set is ready, use cross-entropy. Class weights are fixed as the inverse of the train frequency. The denominator of the sample-weighted mean is the sum of the weights. A negative with no valid target is excluded from target, collision, boundary, relation, and keep. Detection is trained if a source answer exists. State the loss masks for positives and negatives explicitly.

Every auxiliary term averages the valid items among objects, pairs, or relations, then averages the valid samples. If there are zero valid items or samples, that loss is a zero connected to the gradient graph. Do not divide by zero. The total loss at the status stage is L_D+λ_T L_T+Σλ_aux L_aux. Do not replace a sample that has no target with a fake no-op.

Evaluation reports the distribution that is actually used, and records per-class recall and macro-F1. The initial functional gate is macro-F1≥0.90, acceptance of a valid request ≥95%, and recall of each negative class ≥95%. An ok from status does not replace a geometry safety approval.

## 14. Evaluation protocol

### 14.1 Prediction and source identity

Inference uses only RGB and text. Keep only queries with objectness ≥.5. If the predicted count differs from the ground truth, the count fails. Compute the Hungarian match from the source prediction, and check that each match has center error ≤10 mm and rotated-envelope IoU ≥0.5. Do not use the target position in that match. Use this match as it is for target evaluation.

A count or match failure fails the whole scene. Source and target P95 are computed on objects whose match was confirmed, and the match coverage and the number of failed scenes are reported with them. Do not present a success rate that excludes detection failures as the final performance.

### 14.2 Geometry and the instruction

Target safety evaluation uses **ground-truth a and b plus the predicted target pose**. It requires every object pair to have polygon distance ≥10 mm, table-boundary margin ≥10 mm, and overlap area ≤1e-12 m². The distance tolerance is 0.1 mm. This evaluation scores a training result under oracle size. It is not an execution safety check.

Sort only the selected objects by descending y, and use the existing greedy row test so that max y−min y inside a row is ≤10 mm. A row's representative y is the midpoint of min and max, and a member's error is ≤5 mm. Inside a row, x increases. Overall, rows run from top to bottom.

- Requested orientation error ≤5°.
- Adjacent horizontal edge gaps and row-band gaps are the requested value ±5 mm, plus the numerical tolerance.
- Per-color row purity, contiguity of rows of the same color, and 2g between groups.
- Size order is bounding-rectangle area, and an adjacent inversion must not exceed 5%. With color groups, check inside each group.
- The envelope of a selected object must lie inside the requested region.
- A locked object in a partial request is within 3 mm and 3° of the ground-truth source.
- No-op checks the keep condition of every object together with that instruction and the safety conditions. Keep-all does not apply an unnecessary row condition.

The ground-truth dataset validator requires an **exact match** of a locked pose. Do not apply the network evaluation's 3 mm / 3° tolerance to dataset generation. Do not apply M5, which checks invariance of source assets and labels, unchanged to a prediction. Implement a prediction adapter that uses the same geometry and semantic predicates.

**Partial-refinement evaluation scope:** apply orientation, row, size order, color grouping, the requested gap, and the requested region only to the selected set S that was fixed on the current input scene. Apply keep to the locked set K. Apply object preservation and safety geometry to the whole S∪K. Do not pick "the leftmost object" again on the predicted target and change what this sample is scored on. Ground again on that turn's source only when a new instruction is interpreted on the next turn. Do not fail a partial only because locked objects remain scattered, and do not require a full-scene knolling.

Partial task success is the joint satisfaction of the semantic, preservation, and safety conditions above. Not every selected object has to move, and a small movement is not by itself a failure. Coordinate agreement with the teacher target, and J_move, are separate imitation and movement-efficiency metrics. The minimum movement inside the finite candidate set in section 6.6 is the teacher's selection criterion. It is not a condition that forces the network's task success to use the same candidate or the same coordinates. A prediction that fails the semantic or safety check is not accepted as a success because its movement cost is small.

Automatic success and the quality of a training case are also separate. A case with |S|=1 teaches referent selection, replacement or orientation edit, and preservation of locked objects. It is not evidence for learning alignment and gap among selected objects. Report success rate and sample count separately for multi-object edits with |S|≥2, for scattered sources, and for sources that are already arranged. If there is no arranged-source case, mark that ability as not evaluated. Manual QC of partial tidiness uses the requested result on the selected set. Preservation of locked objects and unnecessary movement are scored separately. Do not make an improvement in the tidiness of the whole scene a required condition, and do not judge a case to be a good training example from geometric validity alone.

### 14.3 Reported metrics

| Metric | Definition |
|---|---|
| Count accuracy | scenes with the correct predicted count / all scenes |
| Source errors | median and P95 of position Euclidean mm, size max(\|Δa\|,\|Δb\|) mm, and π-periodic angle in degrees |
| Target errors | median and P95 of target position and angle with the source match held fixed |
| Raw geometry success | count and source match succeed, and the whole gap, collision, and table boundary pass |
| SSR | fraction of scenes that pass count, match, geometry, requested region, orientation, order, grouping, gap, and keep |
| Performance by condition | SSR and n by N, shape and size, color composition, orientation, gap, region, English instruction type, and full, partial, and no-op |
| Language sensitivity | whether each command of a pair on the same RGB succeeds, and whether the crossed instruction conditions are actually distinguished |
| Keep violation | fraction of partial scenes in which any locked object exceeds the criterion |
| No-op | success rate of preserving every object, satisfying that instruction, and passing geometry |
| Refinement diagnostic | error and geometry violations at stages 1, 2, and 3; the final stage does not have to improve monotonically on every scene |

A pair with no real crossed-command difference is excluded from the language-sensitivity denominator, and the count is stated. The final-test report includes a family-level bootstrap confidence interval or the success count over the total. A manual review of the 100-scene QA does not replace a performance test.

The baseline is keeping the source as it is, the same neural model with a fixed default instruction, and the teacher reference. The teacher is an oracle reference that uses the ground-truth state and the structured instruction. Do not claim it as a competing model under the same RGB-only input condition.

## 15. Simulation and the extension to real execution

The first simulation after training moves objects to the predicted targets and checks the layout. That is not manipulation success. The next stage executes with a fixed grasp, lift, and place primitive of a physics simulator and measures drops, collisions, and unreachable placements. The initial object order is a deterministic order of the observed source. If needed, a controller that uses a temporary buffer is verified separately. The controller does not change the network's final target.

Order dependence when source and target overlap in the same plane, and grasp feasibility, are not guaranteed by the absence of collision in the final layout. An execution failure is recorded separately from a layout failure. A 3-turn evaluation feeds the image re-observed from the real execution or simulation, and the next explicit instruction. It does not reset the scene to the teacher answer in between. An episode succeeds only if every turn succeeds.

Inference has no ground-truth size, so the predicted envelope is expanded conservatively and checked before execution. On a separate calibration validation, and only for a correct detection, measure the Hausdorff error between the predicted source envelope and the true envelope. The initial estimate of the perception margin δ_perc is the 99th percentile of the per-family maximum error plus 1 mm. Inflate the predicted footprint by the sum of the calibration error δ_calib and the nominal grasp and place error δ_exec. Recalibrate when the object count, the domain, or the camera changes.

This empirical margin is not a probabilistic safety guarantee against a missed object, a failed grasp, or out-of-distribution input. If the inflated footprint does not pass the shared minimum gap and the table boundary, stop execution and observe again. Do not repair the coordinates with a planner. Checking the natural-language selected set and the requested region against ground-truth semantic fields is possible only in offline evaluation. Do not assume that the real inference safety validator was given hidden ground-truth meaning or a separate parser.

The initial requested gap for introducing a real robot is verified from 30 mm. Reachability, grasp space, table calibration, and an emergency stop are checked separately. Choosing the robot hardware and the physics simulator is not a prerequisite of the current M0–M8 or of the neural pilot.

## 16. Pilot re-verification and the bar for moving on

### 16.1 What must be re-verified

Re-evaluate the existing 130 samples on fixed inputs with the v1.3 teacher, and make a table that compares the new and old targets. The automatic checks must pass on every sample of the new manifest. Run all of the existing 39 regressions and the new v1.3 checks. The criterion is preservation of the meaning, geometry, and reproduction contract, not the test count itself.

Additional fixtures include: both the balanced and the existing partition when both are needed; N=8, vertical, and unbalanced colors; a partial whose center is blocked; English wordings with the same meaning; a change of object array order; no-op reapplication that keeps the same selected and locked sets; a position-based reference whose referent stays or changes when it is reinterpreted; equality of legacy ±π/2 and exact copy of locked and no-op poses; legacy-policy reproduction; and refusal of a mixed policy. When the referent of a position reference changes, check whether the new meaning is satisfied. Do not force no-op. The hash of an unchanged source file and the existing split must be kept. Do not mark every additional fixture as implemented and passed just because this document was corrected.

A person reviews all 130 samples as natural language plus source plus target. Provide images with and without the QC overlay. Score tidiness, whether the command is satisfied, whether a partial move looks natural, and observability, separately. Do not require a tidiness improvement for keep-all. Checking nine pairs on a contact sheet is not a full review.

### 16.2 QA gate

- Zero geometry, identity, instruction, coordinate, mask, and split errors on stored v1.3 samples.
- 100% consistency of repetition, array order, sentences with the same meaning, and no-op under the same selected set, locked set, and requested conditions. Reinterpretation of a position-based reference is judged separately under section 6.2.
- The existing 100 sources are preserved. State the number of changed targets and the number of re-renders.
- A full human review of all 130 is done. At least 90 of the 100 primary samples have an applicable overall quality score of 4/5 or higher. The extra 30 samples are reviewed individually.
- An obvious problem at 2/5 or below is not left with its cause unresolved. The training-approval manifest includes only samples that passed the hard contract and received a human acceptable judgment.
- A repeated quality defect by condition is fixed in the teacher rule. It is not hidden by redrawing only the good cases. If there is no defect, do not add a new aesthetic optimization or a new layout-family experiment.
- A fixed input that fails to generate under v1.3 is kept as valid legacy material and reported as a failure in the new policy's completion rate. Do not mark a complete migration as done unless all 130 succeed. If a training subset is used, state the exclusion list and the coverage by condition.

J_shape does not guarantee a better appearance, so compare v1.2 and v1.3 on the same source and instruction. For partial, also report whether the movement cost decreased inside the same candidate range, and evaluate the arrangement of the selected set and the preservation of locked objects as in section 14.2. Do not require the whole scene to look tidier when a valid result leaves scattered locked objects where they were. If the applicable quality gets regularly worse, review only a single change to the teacher's selection rule. Do not compensate a wrong answer with the network loss.

### 16.3 Gate for moving to the neural implementation

Once the QA gate and a single-policy manifest are ready, implementation of the dataset adapter, model, matching, and loss can start. S0 uses reviewed train samples. Do not claim S2 generalization performance from 100 scenes alone. Proceed to S1 and S2 on the planned amount of data after the CLI limit of 100, chunk and resume, and the expanded split manifest are solved.

The [2026-11-03 final dataset quality review](final-dataset-quality-review.md) separates the assistant's review of all 130 English samples from the automatic audit. An S0 suitability recommendation is separate from a human quality approval. It does not replace the existing human-review gate or the training-approval manifest. Do not regenerate an answer only because of a minor style or placement preference.

The initial goal is to learn a correct match and a target placement for 2–4 objects from detection plus target alone. If that stage fails, diagnose in order: input, units, and answer wiring; then detection; then response to the text condition; then the refinement gradient; then data coverage. Do not turn on collision, relation, and status together while S0 or S1 is failing.

## 17. What changed from v1.2 to v1.3

| Area | v1.2 / current implementation | v1.3 |
|---|---|---|
| Project state | plan-centered; M0–M8 finished later | integrates finished results with the neural and interactive scope that is not implemented |
| Teacher candidates | k=1..4, sequential chunks | k=1..N_s, sequential plus balanced, duplicates removed |
| Full selection | row count, then area, then width | J_shape, then area, then row count |
| Partial selection | translation closest to the region center | J_move first, including the existing-row candidate |
| Geometry, color, and no-op | the corrected v1.2 contract | kept |
| Data handling | generate everything from a seed | preserve the original, migrate fixed sources, render only changed targets |
| Schema | 1.2 | structure stays 1.2; policy and migration sidecars are added |
| Legacy tests | regenerate and compare with the current teacher | per-policy reproduction checks; the existing independent audit stays |
| Language | a restricted legacy Korean-and-English grammar | active data, user examples, and evaluation are English-only; multilingual support is optional later work; meaning is unchanged; counterfactuals are cross-checked |
| Input length | 128 tokens in the earlier network design | no silent truncation; maximum 512 and dynamic padding |
| Network core | object decoder plus 3-stage arrangement | kept; unspecified details are fixed as the section 10 defaults |
| Initial loss | a plan to apply many objectives at once | detection plus target only |
| Matching | source-based identity | kept; the cost and reuse across stages are stated |
| Loss scale | a risk of tuning individual weights | fixed normalization of 10 mm / 5 mm / 10° after restoring meters |
| Auxiliary loss | used from the start | added one at a time when a failure condition is met |
| Status | planned as a full feature | inactive on the positive pilot; negative training is a separate stage |
| Evaluation | geometry, instruction, and preservation | kept; dataset exactness is separated from prediction tolerance; coverage, policy, and execution metrics are stated |

Outside this table, the core scope stays: object count and dimensions, coordinates, gaps, safety conditions, color-pure rows, source–target identity, and the split rule. If an explicit v1.3 rule conflicts with an older document, v1.3 applies to later implementation, and the provenance of the older output remains.

## 18. Minimum-change checklist

### A. Changes required in the current generator

- [ ] Separate the M4 policy explicitly into `legacy_v1_2` and `balanced_rows_v1.3`.
- [ ] Implement the wider k, both partitions, duplicate removal, J_shape, J_move, and the deterministic tie-break.
- [ ] Reflect the movement-first comparison of the partial existing-row candidate and the full candidate set.
- [ ] Preserve no-op, color purity, locked poses, the real size and orientation calculation, and the distinction among failure states.
- [ ] Make the M3 row-band wording explicit, and confirm that the structured meaning and the numbers did not change.
- [ ] Add a path in M7 for fixed-input migration that updates only a changed target.
- [ ] Add policy, determinism, counterfactual, and quality and movement comparison reports to M8.
- [ ] Separate the legacy reproduction check from the new-policy check, and do not lower the existing audit bar.

### B. Work to redo on the existing 100 scenes

- [ ] Confirm that the original manifest, source, asset, and split are preserved.
- [ ] Compute the teacher difference on 130 samples and write the change list.
- [ ] Render only changed targets, and update the related annotation, relations, hashes, and QC together.
- [ ] Include unchanged images in the full audit of the new manifest.
- [ ] Check the crossed instruction conditions of the 30 counterfactual pairs.
- [ ] Record a full human review of 130 samples and the quality distribution by condition.
- [ ] Report the old acceptance rate and the migration success rate separately.

### C. Changes required only in the neural implementation

- [ ] Build an adapter that separates an existing annotation into an RGB-plus-text input and supervision labels.
- [ ] Check token length, coordinates, angle, the size head, and query padding and masks.
- [ ] Implement a 4-layer object decoder and a 3-stage arrangement, with the same query identity.
- [ ] Implement one source-only Hungarian match, and supervise every target with that same match.
- [ ] Implement the initial L_D+L_T, the fixed scales and weights, and a switch for conditional auxiliary losses.
- [ ] Confirm that ground-truth size, selection, and lock do not enter the inference input.
- [ ] Implement the prediction adapter and the raw-geometry, SSR, keep, and language-response metrics.
- [ ] Apply the gates in the order S0, then S1, then S2, and add an auxiliary loss after diagnosis.

Sequential-edit data, negative status data, and robot execution are in scope only after the basic neural pilot in C passes. The whole of the already finished M0–M8 does not need to be rewritten.

## 19. Conditions for considering this document applied

The deliverable of this revision itself is the integrated specification, the change table in section 17, the minimum-change checklist in section 18, and the re-verification gate in section 16. Claiming that the later implementation is done requires the new teacher policy, the migration, and a full QA result. Claiming neural success requires a separate result on a fixed test. In the currently approved scope, only the document is updated. The code, the dataset, the Git history, and the training state are not changed.
