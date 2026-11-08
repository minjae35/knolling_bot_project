# recovery_v3_joint — implementation review

Implemented on `feat/neural-recovery` for independent review before training. No
Knolling model optimizer update, performance experiment, or new training process
was run. Both v3 configurations retain `execution: review_only` and
`allow_training: false`. The old AR experiments and checkpoints remain intact.

## Design-to-code mapping

| Requested design | Implementation and review entry point |
| --- | --- |
| Joint, permutation-equivariant target poses | `src/knolling_model/recovery/joint_layout.py`: shared per-object projections, object self-attention and language cross-attention; no object index embedding, causal mask, target sorting, or AR prefix. Output rows follow input objects. |
| Two weight-shared refinements | One `refinement` block, one `state_projection`, and one `update` head are reused twice after the initial prediction. Continuous displacement, softmax bin probabilities, and tanh residuals feed the next refinement without detach or argmax. |
| RGB + language inference | `recovery/model.py`: unchanged RT-DETR source prediction and label-free query selection feed the new decoder. Frozen MiniLM supplies token memory. `forward` accepts only images, token IDs, and attention mask. No teacher/planner import or target input. |
| Final-only three-term loss | `joint_pose_loss`, dispatched by `recovery/stages.py`; no intermediate loss, Move/Keep head, target copy gate, geometry loss, relation loss, or layout semantic classifier. |
| New teacher and annotation overlay | `knolling_dataset/teacher_v3.py`, `recovery/dataset_v3.py`; offline teacher with pinned code/config and replay-validated annotations. |
| Independent task evaluation | `recovery/evaluation_v3.py`; actual geometry/semantics and command contrasts determine validity, while teacher-coordinate fidelity is reported separately. |
| Checkpoint/resume and stage gates | `recovery/driver.py`, `recovery/stages.py`; v3 data and evaluator dispatch, architecture/schema binding, existing bounded execution and rollback logic. |

The default legacy factory remains `recovery_v2_ar` to preserve historical
configurations. **The new review/execution configs explicitly select
`recovery_v3_joint`.** Legacy AR weights are not partially loaded into the new
decoder. Reused base weights are the cached RT-DETR R18 and MiniLM assets.

## Architecture and loss details

Default width is 128, with four attention heads and two initial joint attention
blocks. Each refinement contains self-attention, language cross-attention and a
feed-forward layer. Object features retain source XY, size, periodic angle and
soft color/shape probabilities. Object/query IDs are bookkeeping only.

An always-valid learned null self-attention key handles empty detections; invalid
rows are masked. Final target coordinates are not clipped or repaired. Each
refinement adds a bounded displacement correction of at most 50 mm per axis;
the initial displacement is unbounded. This bound is an implementation choice
for Work to review, not an experimentally established optimum.

For final displacement `z`, source position `s`, and teacher position `t`:

```
predicted_position = s + 0.010 * z
L_position = mean_scene mean_valid_object sum_xy Huber((predicted_position - t) / 0.010)
L_angle_CE = mean_scene mean_valid_object CE(final_bin_logits, teacher_bin)
L_angle_residual = mean_scene mean_valid_object Huber(tanh(final_residual[teacher_bin]) - teacher_residual)
L = L_position + L_angle_CE + L_angle_residual
```

Huber is PyTorch SmoothL1 with beta 1. Position normalization is therefore
equivalent to supervising source-relative displacement in 10 mm units. The
angle representation reuses `PeriodicBins`: 18 bins over pi and a normalized
within-bin residual. Bin decoding uses argmax only at the final output, never
inside refinement. The loss uses logits and the teacher-bin residual directly.

Every valid object contributes, including locked/Keep objects. Padding does not
contribute. Loss is averaged within each scene and then across scenes; larger
object counts do not implicitly increase scene weight. Three states (initial,
refinement 1, refinement 2) remain available for gradient diagnostics, but only
the last state's pose is supervised. Pose heads/loss arithmetic use FP32.

These three terms are the complete **Arrangement pose loss**. Stage 2 retains
the existing detector/source supervision, and the optional joint phase retains
the existing independent unique-source detection replay loss. These are not new
layout semantic/geometry auxiliaries. Frozen MiniLM receives no gradients.

## Dataset and offline teacher

Active annotation file:
`artifacts/recovery_v3_verification_r1/annotations.json`

SHA-256:
`b42ee5b9e99afeeb37036a63f4e180b791719212faa68be226f7745beb051e7c`

- **25 existing source families, 76 unique annotations, 35 verified contrast pairs.**
- Pairs: axis 21, region 5, selection 6, gap 1, sort direction 1, grouping 1.
- All sources/RGB are reused; no new scene rendering or large dataset expansion.
- Existing 96 legacy/control records are relabeled under one teacher and
  deduplicated. Aliases preserve their relationship to the original sample IDs.
- The three additional contrast types reuse existing sources. Rejected candidate
  contrasts and reasons are recorded; every accepted pair changes only its
  declared factor and has a bidirectional semantic effect.
- This is a **train verification suite**, not a held-out generalization benchmark.

`continuous_rows_v3` first preserves an already valid layout exactly, including
valid no-op cases. Otherwise it enumerates deterministic row templates, excludes
observed-source templates, and uses continuous feasible translation instead of
a translation grid. Full-layout ties use shape/area/row/template criteria;
partial-layout candidates prioritize movement while preserving locked objects.
Numerical tie comparisons use 12 decimal places. All targets pass a separate
validator. Exhausting the conservative candidate family is explicitly a
`generation_failure`, not proof that the task is infeasible.

Teacher acceptance uses axis 2 degrees and gap/row tolerances 2 mm, aligned with
the v3 evaluator rather than the looser legacy defaults. Teacher locked poses
are exact copies. Geometry retains the existing 10 mm clearance and workspace
margin with the validator's numerical epsilon.

Loader checks include annotation hash, teacher code/config hashes, parent
manifest/selection, family/source/split equality, RGB bytes, language audit,
teacher replay, geometry, derived metadata, duplicate observable inputs and
contrast coverage. `copy_required`, exact/task movement, structure center and
neighbor metadata are annotation diagnostics; they are not model inputs or new
loss targets. Source/target annotations remain supervision/evaluation data.

An earlier prototype file under `artifacts/recovery_v3_verification/` is retained
but **is not selected by either config**; its old teacher provenance will not
pass the current loader. The active release is the `_r1` file above.

Rebuild into a **new directory** using the existing parent/control suite:

```bash
conda run --no-capture-output -n knolling python -m knolling_model.recovery.dataset_v3 \
  --config configs/recovery_gpu_initial.json --output /tmp/knolling-v3-fresh-review
```

The rebuild test requires byte-identical annotation SHA-256. Existing directories
are rejected, and the parent dataset is never an output destination. Annotation
artifacts stay outside Git; the configs, generator and hashes are committed.

## Evaluation and execution contracts

Teacher fidelity reports matched target P95 position/angle errors and a separate
1 mm / 2 degree criterion. Task validity checks source correspondence, collision,
clearance, workspace/region, Keep, axis, row alignment, requested gaps, grouping
and sort order. Keep tolerance is 1 mm / 1 degree; axis is 2 degrees; gaps and
row deviations are 2 mm. Size ordering retains the existing 5% tolerance.
Physical evaluation uses ground-truth footprints after source correspondence;
those annotations never enter prediction.

A geometrically and semantically valid alternative target may pass task validity
while failing teacher fidelity. Move accuracy remains diagnostic, derived from
pose change greater than 3 mm or 3 degrees; it is no longer a classifier score.

Proposed v3 gates, pending Work's review:

1. `oracle_layout`: complete pinned suite, every task valid, all six contrast
   factors covered and all verified pairs successful.
2. `perception`: complete suite, exact count/coverage, zero FP/FN, correct
   color/shape, source position/size P95 <= 1 mm and angle P95 <= 2 degrees.
3. `predicted_layout`: perception criteria plus task validity and contrasts.
4. `joint`: the preceding criteria plus full-suite no-regression versus its
   entry baseline. Regression covers source metrics, task/Move accuracy,
   contrast success, Keep violations and per-mode task success.

The two latter phases are the existing driver's internal Stage 3 sequence.
Teacher fidelity is reported independently and does not reject a valid
alternative solely for differing coordinates. Missing samples/pairs, wrong
schema/source mode, overflow and unsupported stages cannot silently pass.

Checkpoints bind code commit, full config, data fingerprint, architecture and
evaluation schema. Model loading is strict. Interrupted same-stage runs can
resume with optimizer/RNG/cursor; terminal budget exhaustion is not relabeled
as an interruption. V3 explicitly rejects the old Stage 1 extension contract.
The existing timeout final evaluation and joint regression rollback are retained.
Execution still requires an enabled config, reviewed receipt, clean checkout and
passed preceding stage. **No training receipt was issued in this task.**

## Verification evidence

All Python executions used the `knolling` environment. Logs and preservation
evidence are in `artifacts/recovery_v3_review/`.

| Check | Result |
| --- | --- |
| Static Python syntax / whitespace | 114 Python files parsed; `git diff --check` passed |
| Complete CPU regression suite, including cached recovery base model | **234 passed, 24 skipped**, 59.44 s |
| Final evaluator/checkpoint integration recheck | **7 passed, 2 CUDA checks skipped**, 15.75 s |
| Explicit RTX 4090 CUDA checks | **2 passed**, 8.81 s |
| Original artifact preservation | 2,189 prior files retain size and modification time; old final Stage 1 checkpoint SHA-256 unchanged |

CPU skips are explicitly gated CUDA, Blender/rendering, migration/deliverable and
separate legacy pretrained integrations; they are not claimed as tested.

Specific coverage:

- Every permutation for 1–4 objects, 20 permutations each for 5–8; position
  tolerance 1e-5 m and angle tolerance 1e-4 rad. Query-ID changes do not affect poses.
- Object/text padding, finite empty detections, variable object counts; no
  order embeddings or per-object-index parameters.
- Exactly three loss keys; explicit 10 mm normalization; Keep-only batches;
  nonzero finite gradients through all three states, language memory and source
  physical attributes. No intermediate loss or recurrent detach.
- Training/inference output equality with identical selected sources; modifying
  target/Move labels does not change predictions. Target arguments are rejected
  by the decoder interface.
- All 76 targets replay exactly; reversing source-object order preserves teacher
  target poses. Rehashed corrupt target/text/teacher/RGB/split/metadata/pair files
  are rejected.
- Synthetic exact evaluator fixtures pass gates; incomplete coverage, ignored
  counterfactual instructions and collisions fail. A valid shifted arrangement
  passes validity while failing fidelity. These fixture outcomes are **not model
  accuracy measurements**.
- Zero-update real joint-decoder checkpoint round trip; incompatible AR states
  and mismatched checkpoint contracts rejected. Pause/resume optimizer/RNG/cursor
  equivalence uses a scalar fixture only, not Knolling training. Existing timeout
  and rollback tests remain in the full CPU suite.
- CUDA tiny RT-DETR runs all four stage forward/backward paths, including layout
  gradients into RGB/perception. Cached real R18 + MiniLM runs batch-4 oracle and
  perception forward/backward plus RGB inference. All model weights/buffers are
  checked unchanged. Peak allocated memory: oracle **624.02 MiB**, perception
  **2,844.58 MiB**. These are verification peaks, not training memory estimates.

Old final Stage 1 checkpoint SHA-256 remains
`a5346ad7c9f8cd6c9aed14c5e64c6c76319802612d87fb5b9ef98b6ad0024984`.

## Remaining limitations and Work review priorities

1. Review the three-term loss reduction, periodic residual convention, source
   scaling and shared-refinement graph against the final design. The 50 mm
   refinement correction bound and equal term weights have not been trained.
2. Review teacher canonicalization and partial-layout policy. Continuous
   translation is still conservative about the arrangement envelope; it is not
   a complete solver. Twenty-five families cannot establish policy smoothness
   or broad generalization. Gap/sort/group contrasts have only one pair each.
3. Review task-success gates separately from fidelity. No Keep hard-copy gate or
   geometry repair exists; loss alone does not guarantee valid predictions.
   Verify the 1 mm Keep and 2 mm gap/row tolerances fit the intended task.
4. Review inference query selection, target-only matching, detection replay and
   source-quality prerequisite. The real pretrained detector is not a trained
   knolling detector; complete real R18 end-to-end joint backward at learned,
   correct detection counts has not been demonstrated. Tiny-detector tests
   establish graph wiring, not final detection quality.
5. Review resume/schema binding and stage transition logic before issuing a new
   receipt. Old AR checkpoints and the old budget-extension receipt do not
   authorize or initialize v3 learning. Proposed config budgets are placeholders
   for review, not authorization to train.

Implementation commits: `f66fdd0` (joint model/loss), `47f7855` (teacher/data),
`cd43792` (evaluation/driver/tests). Commit author and committer use the existing
repository identity; pushes use the configured personal SSH key.
