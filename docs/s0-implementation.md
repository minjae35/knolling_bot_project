# S0 implementation and launch contract

Status, 2026-11-03: S0 components are implemented and locally verified. **No
S0 model training, GPU training or GPU provisioning has been performed.**
CPU regression tests use small synthetic models with real optimizer steps.
The user confirmed the final dataset quality review is complete and authorized
S0 preparation. Historical dataset approval sidecars remain unchanged. Starting
training awaits review of this implementation.

The architecture and learning strategy follow
[Technical Specification v1.3](technical-specification-v1.3.md), sections 10–14.
This document fixes implementation details and commands; it does not expand S0
into S1/S2, auxiliary geometry losses, execution planning or robot control.

## Implementation interfaces

| Module | Contract |
|---|---|
| `knolling_model.data.S0Dataset` | Active manifest lookup only; verifies pinned manifest, selected annotation/source-RGB hashes, English policy, Train split, N=2–4 and source–target identity. Returns RGB/text separately from labels and metadata. |
| `Collator` / `Batch` | `inputs` contains only `images`, `input_ids`, `attention_mask`. Supervision and metadata have separate fields. Every model call uses `model(**batch.inputs)`. |
| `VisualBackbone` | RGB float32 `[B,3,384,576]`, /255 in adapter, ImageNet normalization inside model. ResNet-18 with FrozenBatchNorm; FPN strides 8/16/32, 256 channels, 4,536 image tokens plus 2D/level positions. No resizing or augmentation. |
| `KnollingModel` Object Decoder | 12 learned queries; four pre-LN blocks, eight heads, FFN 1,024/GELU; image cross-attention; `[B,12,256]` object tokens. |
| Source heads | Objectness `[B,12]`; uv and a/b `[B,12,2]`; angle `[B,12]`; color/shape logits `[B,12,3]`. Positive sizes with a≥b; initial a/b = 60/35 mm. π-periodic orientation. |
| Text path | Pinned MiniLM token encoder, frozen weights and eval mode; 384→256 trainable projection. English inputs only; dynamic padding, ≤512 tokens including special tokens, no truncation. No external instruction parser. |
| `ArrangementDecoder` | Three self/text-cross-attention blocks. Same query retains the same identity at every stage. Geometry embedding uses predicted source and previous target only, with no detach or GT masks. Each output contains target uv `[B,12,2]` and angle `[B,12]`. |
| `S0Criterion` | One source-only Hungarian assignment per scene; reused for source losses and all target stages. Returns scalar batch loss, individual loss terms and assignments. |
| `evaluation` | Threshold q≥0.5 first, then source correspondence checks, per-instance errors and three-stage geometry/semantic diagnostics. Evaluator GT never enters model inference. |
| `runner` | Seeded family sampler, FP32 AdamW, scene-weighted accumulation, checkpoints/resume, per-epoch evaluation of all 32 instructions. Preflight has no optimizer. |

`decode_predictions` returns query indices and their source/target predictions
after the 0.5 threshold. There is no GT top-k, NMS, coordinate repair or separate
layout planner. Status/Keep/Collision/Relation losses and an execution adapter
are not implemented in this S0 milestone.

The three residual heads start at zero, so initial targets copy predicted
sources. Center residual step factors are `(1, .5, .25)` in logit-uv space;
angles use `(π/2)tanh(residual)` followed by π-periodic wrapping. The first
backward pass can legitimately have zero gradients in layers behind those zero
final weights. Tests separately verify that target gradients reach earlier
refinements, source geometry, image backbone and text projection once residual
weights are nonzero. This is not evidence that convergence has been achieved.

## Coordinates, matching and losses

Workspace x/y are in meters with origin at bottom left and dimensions 0.6×0.4 m.
`xy = uv * (0.6, 0.4)`; pixel coordinates use the documented half-pixel offset
and invert y. Images are not vertically flipped. Heads produce normalized
centers, physical a/b lengths, and radians; losses convert centers to meters
before comparing to labels. GT target rows are joined to source objects by
`object_id`, never by annotation-list position.

For source query i and GT object j, matching cost is:

```text
-sigmoid(q_logit_i)
+ L1(source_xy_i - source_xy_j) / (2 * 0.010 m)
+ L1(size_i - size_j) / (2 * 0.005 m)
+ abs(periodic_angle_delta_i_j) / (π/18 rad)
```

SciPy Hungarian matching runs outside the gradient graph. Target pose, class,
selection and lock labels do not influence matching. All matched GT objects
receive target supervision, including locked objects and queries below 0.5.

Loss definitions implement v1.3 §11.1–11.4 directly. Smooth-L1 has beta=1 and
uses 10 mm, 5 mm and 10° error scales for position, size and angle. Position/size
first average the two coordinates, then objects within a scene, then scenes.
Objectness BCE weights positive queries 1 and unmatched queries .1, normalized
by `N + .1*(12-N)`. Detection adds source position/size/angle and .1 times each
color/shape cross-entropy. Target position+angle losses use stage weights
`(.25, .25, .5)`. Total loss is Detection + Target. Stage loss weights are distinct
from the residual step factors.

## Fixed S0 selection

[s0_samples.json](../configs/s0_samples.json) pins all 32 IDs and the manifest:

```text
8000428a04522f50cf37567d85d442ac94d1f8a87121747d4243e33974320d06
```

The selection contains 16 Full, 8 Partial and 8 No-op samples, all Train and all
with 2–4 objects, across 25 source families. It includes seven complete
same-image counterfactual instruction pairs, both eligible multi-object Partial
examples (two selected objects each), six single-object Partial examples, three
Keep-all and five repeated-request No-ops. The exact deterministic selection
rule is recorded in the configuration. It does not filter on model performance.

These examples are suitable for a wiring/memorization diagnostic, not proof of
multi-object refinement coverage or language/geometry generalization. No split,
instruction, image, target or Teacher output was edited. The fixed grammar is
retained; measured lengths are 13–143 tokens with the actual pinned tokenizer.

Family-uniform sampling visits each of the 25 families once per epoch. Within a
family it cycles through a shuffled instruction list, so all 32 instructions
are seen over two epochs. Every evaluation includes all 32, not just that
epoch's 25 visits. Train examples themselves determine the overfit gate; held-out
Validation/Development/Test samples are not used to tune this memorization run.

## Training and checkpoints

[s0.json](../configs/s0.json) sets seed 1203, FP32, dropout=0, no augmentation,
fixed learning rates, maximum 2,000 optimizer updates, gradient clip norm 1.0
and weight decay 1e-4. Pretrained `backbone.body` parameters use LR 1e-5.
New FPN/position embeddings, decoder/heads and text projection use 1e-4. All
biases, including attention projection biases, and LayerNorm parameters have
zero weight decay. Frozen text parameters are excluded from the optimizer.

Default micro batch is 2 and effective batch is 16. Each family epoch produces
one 16-scene update and one 9-scene tail update. Loss accumulation weights each
micro batch by its actual scene count divided by the update's scene count; the
one-scene tail is not overweighted. There is no LR scheduler or AMP in S0.

Python/NumPy/torch seeds and RNG states are recorded; deterministic algorithms
are required, TF32 and cuDNN benchmark are disabled. A nondeterministic operation
raises rather than silently weakening this setting. Exact reproduction across
different CUDA/driver/hardware versions is not promised.

Each run uses a **new** output directory. `metrics.jsonl` records update losses,
gradient norm, effective scene count and per-epoch evaluation summaries.
`evaluation-latest.json` includes all sample failures and per-stage errors.
`last.pt` is saved every 50 updates and at normal completion. For S0,
`best.pt` first admits memorization-gate-passing states ahead of failing states,
so a successful checkpoint cannot be displaced by a failing state with better
aggregate diagnostics. Within each gate class, selection follows v1.3 §12.1:
Task Success, then Raw Geometry, then lower target-center P95 error. This is
selection on the 32 Train samples, not held-out model selection. Run completion
writes `result.json`.

Checkpoints contain trainable model parameters, visual buffers, Adam state,
RNG states, the next epoch/group cursor, configuration, source-code hash, Git
revision/dirty status, input manifest/selection hashes, loss scales and token
settings. Frozen text weights are omitted and reloaded from the pinned revision
in [pretrained.json](../configs/pretrained.json). Keep that cache available.
The resolved dependency freeze and device information are saved with the run.
Resume rejects changes to source code, configuration, dataset selection,
pretrained pins, language/Teacher policy or loss/token settings. It writes into a
new directory and records its parent checkpoint hash, preserving prior logs.
Resume evaluates the restored weights before any optimizer update. A successful
S0 run exits immediately with unchanged weights, optimizer, RNG and cursor; an
interrupted run continues from the exact next group. The extra evaluation
preserves the restored RNG stream. Each new directory selects `best.pt` among
states evaluated there, beginning with its restored checkpoint; it does not
claim to reconstruct an unavailable historical best from the parent run.
Source-hash validation intentionally rejects checkpoints made by older code.
Start the reviewed S0 run from the new revision rather than editing provenance
to make an old checkpoint load.

Training stops at the S0 memorization gate or 2,000 updates, whichever comes
first. The gate requires count accuracy and confirmed source correspondence
coverage both 100%; source/target center and max(a,b) size error P95 ≤1 mm;
source/target orientation P95 ≤2°. Source correspondence additionally requires
center error ≤10 mm and predicted-source-envelope IoU ≥0.5. Errors from a few
correct detections cannot hide missing objects.

Safety is evaluated with GT dimensions and predicted target poses as an oracle
geometry diagnostic, never as a model input or an inference repair. Raw geometry
requires 10 mm separation and boundary margin. Instruction checks use prediction
tolerances from v1.3; locked and No-op objects use Keep ≤3 mm/3°. Per-mode task
success and residual safety/semantic failures remain explicit even if the
memorization gate passes. A passing S0 gate is not robot execution clearance.

If S0 fails, inspect units/identity/matching and per-loss logs first, then count
and source localization, instruction-pair responses and refinement gradients.
Do not add collision losses to hide a failing Detection + Target diagnostic.

## Commands

Run from the repository root. The Desktop workspace is
`/Users/minjae/Desktop/knolling_bot_project`. The existing isolated `.venv` is
installed; the historical dataset Conda environment was not changed.

For a new local CPU environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-neural-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
```

For future Linux/CUDA setup and an approved transfer of the existing English
pilot, use [the GPU setup runbook](vast-ai-s0.md). Do not blindly reuse macOS
torch wheels on Linux; its setup script pins the CUDA build explicitly. Dataset
transfer/provisioning is a separate action, not performed by this implementation.

Read-only checks, including a real pretrained forward/backward without any
optimizer update:

```sh
.venv/bin/python scripts/verify_s0_data.py
# On the first machine only, allow downloading the pinned public weights:
.venv/bin/python -m knolling_model preflight --device cpu --download-weights
# Subsequent offline preflight and tests:
.venv/bin/python -m knolling_model preflight --device cpu
RUN_PRETRAINED_TESTS=1 .venv/bin/python -m pytest -q tests/neural
```

**After implementation review, on a CUDA host only, the exact launch is:**

```sh
.venv/bin/python -m knolling_model train \
  --config configs/s0.json --device cuda \
  --run-dir artifacts/s0/runs/seed1203 --start-training
```

Resume without overwriting the original run:

```sh
.venv/bin/python -m knolling_model train \
  --config configs/s0.json --device cuda \
  --run-dir artifacts/s0/runs/seed1203-resumed \
  --resume artifacts/s0/runs/seed1203/last.pt --start-training
```

No train command was executed during preparation. The CLI requires the explicit
`--start-training` flag. Preflight defaults to CPU and offline cached weights;
neither command provisions hardware. Output paths inside the input dataset are
rejected. All run artifacts and pretrained weights stay outside Git.

## Local verification and hardware estimate

- Full regression run with all three existing pilot audits and real pretrained
  neural tests enabled: **108 passed, 5 skipped**. The skips are opt-in Blender,
  builder and migration integrations; no dataset pipeline code changed, and no
  dataset regeneration was requested or performed.
- Final focused neural run: **35 passed**. Coverage includes shapes, physical
  units, identity permutations, source-only assignment, numeric loss fixtures,
  gradients through all refinement stages, padding, no GT input path, perfect
  and failing evaluator fixtures, accumulation, optimizer groups and checkpoint
  RNG/cursor/state restoration. Trainer control flow uses a mocked optimizer;
  it is not an actual training run.
- Actual cached ResNet/MiniLM CPU preflight: batch 2 at full resolution, finite
  loss/gradients, 32 instructions checked, frozen text parameters, unchanged
  trainable weights and **zero optimizer steps**. Its initial random-model loss
  is a wiring observation, not a convergence result.
- `pip check` and Git diff checks passed. Hash comparisons show all 1,041 v1.2,
  1,614 v1.3 and 1,750 English-pilot files unchanged. Local machine reports are
  under `artifacts/s0_verification/` and are not committed.

The network has **138,906,582 total / 21,252,822 trainable parameters**. A single
NVIDIA GPU with **16 GB VRAM** is a conservative starting estimate for FP32,
micro batch 2 and accumulation to 16; 24 GB provides additional margin. This is
an estimate, **not measured CUDA memory or tested GPU compatibility**. If memory
is insufficient, start a new configuration/run with `micro_batch: 1`, preserving
effective batch 16 and scene-weighted tails. Do not change an existing resume
configuration. Keep several GB for caches and checkpoints; the two checkpoint
files are approximately 0.5 GB together once Adam states exist, excluding caches.

CUDA installation, deterministic kernel compatibility, peak VRAM, throughput
and actual 32-sample convergence remain unverified. Measure those on the first
authorized GPU run; no wall-clock training-duration claim is made here.

Verified code milestones: adapter/selection `93594b4`; integrated model
`3a6818d`; matching/loss/evaluation `9b80aa5`; launcher/checkpoints `198c4aa`.

## Review fixes: evaluation diagnostics

The v1.3 §14.3 summaries are additive diagnostics; the S0 gate and model inputs
remain unchanged. Each condition bucket reports sample and success counts,
Task Success, Raw Geometry and failure counts. Conditions include object count,
shape/color composition, exact a×b sizes in mm, direction, gap, region, grouping,
order, English subject type, and Full/Partial/No-op mode. Partial reports split
single versus multiple selected objects and structurally organized versus
scattered sources. The source-layout label reuses the dataset quality audit's
conservative structural test; it does not establish interactive episode history
or human-rated quality. Empty Partial buckets explicitly have `n=0`, null rates
and `evaluated=false`.

Keep violations count Partial scenes with at least one locked object exceeding
3 mm/3°. Count/correspondence failures remain in Task Success denominators and
are marked unchecked for Keep; its scene rate is null when any Partial scene
cannot be checked. No-op success includes all-object preservation, applicable
instruction checks and geometry. All three refinement stages report center and
angle Median/P95, geometry successes, coverage and geometry failure counts.

Language diagnostics pair different instructions within the same source family
and RGB content hash, across all evaluation batches. Each eligible pair must
pass both own-command evaluations and reject both cross-command evaluations.
Pairs whose GT targets do not violate both opposite commands under prediction
tolerances are excluded and counted explicitly. Detection failures remain
failures in the eligible denominator. No pair coverage produces a null rate,
not a claim of language sensitivity. Structured instructions and GT dimensions
are used only by the offline evaluator, never by model inference or the gate.

Review validation uses the isolated Python 3.12 CPU environment at
`/tmp/knolling-s0-review-venv` with pinned neural dependencies and CPU builds of
torch/torchvision. Synthetic tests cover real AdamW updates, exact interrupted
resume, no-update successful resume, checkpoint promotion, diagnostic coverage,
language-insensitive predictions, excluded pairs and JSON report integration.
No S0 neural training or GPU optimizer updates were run.

The English pilot became available during review. Its adapter and oracle
neural tests run, but `scripts/verify_s0_data.py` currently fails the unchanged
full audit with five errors: `teacher_scores_or_configuration` and
`migration_scores` for `scene_0037_primary` and `scene_0089_primary`, plus
`partial_quality_mismatch`. Read-only comparisons isolate the numerical
mismatches to recomputed `J_move`: 4.440863602014575e-06 versus
4.440863602014576e-06, and 0.05376727341039903 versus 0.05376727341039904.
The audit reports no file-hash or target-reproduction failures. This is evidence
of floating-point audit sensitivity; it is not a passing full-data audit. No
Dataset, Teacher, approval sidecar or audit policy was changed. Resolve that
readiness check separately before advancing the GPU launch gate. GPU preflight,
memory requirements and convergence remain unverified.
