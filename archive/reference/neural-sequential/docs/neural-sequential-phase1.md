# Neural sequential Phase 1: fixed implementation and experiment

One candidate on `experiment/neural-sequential-phase1`. The scope is Train791,
Validation94/23 counterfactual pairs, no Test, no Micro repetition, no automatic
retry or model search. Existing datasets, Best registry and main are protected.

## Network and exact training paths

Frozen Perception, MiniLM, Goal, Orientation and Move/Keep come from the same
pinned 2k generalized reference as the Single-Row experiment. Failed Single-Row
XY weights are not copied. Two independent copies of its reference Set encoder
initialize the selector and pose branches; their trainable parameters are
disjoint. This prevents pose loss from directly changing selector parameters,
not gradient interference between scenes inside either branch.

The selector reads the predicted Goal and all observed objects. Attention scores
select an unvisited predicted Move object at each step. Only padding, already
selected objects and predicted Keep are masked. Stable query IDs break exact
score ties; no size sorting, Teacher order, collision test or planner is used.

The pose branch attends to all observed objects, their predicted partial layout,
remaining objects and predicted Keep objects. A neural head outputs a 2D offset
from the previous predicted position (the first reference is the constant
workspace centre). Both X and Y are learned; no common-Y constraint or analytic
spacing formula is present. Predictions are returned in the original ID order.
This can represent multiple rows, but only Phase 1 is trained and evaluated.

Training performs one AdamW step per batch with two losses and independent
parameter groups:

- Selector CE: existing Teacher row/order labels supervise the next object;
  Teacher preceding IDs are used only inside this supervised CE computation.
- Pose: unchanged actual-output position Huber with 10mm scale, using the
  **predicted order and predicted preceding XY**, as at inference. No GT prefix
  trains the pose branch, and no surrogate argmax gradient joins the branches.

The choices are fixed: seed1203, batch16, AdamW1e-4, weight decay1e-4, FP32,
norm clipping1 separately for the two disjoint groups. No additional geometric
loss, calibration or hyperparameter search is performed. Modules are built with
existing PyTorch components; external research code was not copied.

## Fixed evaluation and budget

One GPU run after CPU/CUDA preflight, at most 3000 updates /1200 driver seconds
including evaluation; reserve120 seconds for final replay. Paired Oracle/RGB
Validation runs every250 updates with all94 scenes retained, including count
failures. No interim loss decline is treated as Task success.

Incremental PASS requires **RGB Task >4/94 and CF >1/23 simultaneously**, with
exact frozen-state/output and selected-checkpoint replay checks. This is the
user's comparison criterion, not Phase 1 completion (still94/94 and23/23).
Stop immediately upon comparison PASS, budget exhaustion or numerical/integrity
failure. No automatic restart. Best ranking is comparison-PASS first, RGB Task,
RGB CF, Oracle Task, then lower RGB gap P95. Previous successful scenes lost and
new successes are separately recorded; neither is hidden by aggregate results.

At every assessment, three Oracle paths separate failures:

1. Free rollout: predicted order and predicted prefix; the deployable path.
2. Teacher order: correct object sequence, predicted pose prefix; diagnostic only.
3. Teacher prefix: correct object sequence and previous GT positions;
   one-step diagnostic only, never a model-selection score.

Exact Teacher sequence, correct steps, and exact-sequence scenes failing gap
checks are reported separately from semantic order tolerance. Teacher prefix
Task/CF results must never be presented as RGB or autonomous success.

## Protection and checks

Unit tests use synthetic CPU/CUDA updates for restoration checks. Actual-data
preflight has zero optimizer steps and checks permutation/padding, frozen
gradients, all94 RGB commands on CPU/CUDA, eight-object synthetic inference,
and actual split-checkpoint replay. Original annotation, baseline prediction,
starting reference and evaluator hashes are checked before/after the run.

Checkpoint files contain model, both optimizer groups, RNG, data exposure order,
configuration, update count and elapsed time, using existing atomic split
storage. Keep their shared directory. A unique execution marker and GPU lock
prevent duplicate runs. No automatic training resume is authorized by this run.

Entry point: `scripts/run_neural_sequential.py preflight|train`.
Configuration: `configs/neural_sequential_phase1.json`.
Artifacts: `artifacts/neural_sequential_phase1_once/`.
Use Conda `knolling`, existing offline model caches and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Preflight completed: 11 unit/driver tests passed, including CUDA synthetic
resume. Actual CPU and CUDA checks each covered all94 RGB commands; permutation
and padding differences were zero on the 2/3/4-object probes. Both devices had
finite nonzero gradients into both branches, frozen state unchanged, and exact
checkpoint reload. Actual-data optimizer steps before training: zero.

## Completed single run (2026-11-10 UTC)

**Incremental comparison PASS; Phase 1 completion FAIL.** The run stopped at
2250 updates, the first assessment satisfying both requested improvements,
after 735.61 driver seconds (12 min 15.61 s), including initialization,
Validation diagnostics and final checkpoint replay. Training code: `8e4311e`.
No retry, extension, Test evaluation, subsequent training or Best promotion.

| Independent Validation metric | Prior selected Single-Row | Neural Sequential, update2250 |
|---|---:|---:|
| RGB Task /94 | 4 | **6** |
| RGB Counterfactual /23 | 1 | **2** |
| Oracle Task /94 | 0 | **6** |
| Oracle Counterfactual /23 | 0 | **1** |
| RGB gap P95, mm | 63.311 | **12.811** |
| RGB Teacher XY P95, mm | 111.338 | **124.369** |
| RGB orientation P95, degrees | 0.47822 | 0.47822 |
| RGB gap-failed scenes | 85 | 78 |
| RGB size-order-failed scenes | 57 | 60 |
| RGB collision-failed scenes | 62 | 1 |
| RGB clearance-failed scenes | 70 | 17 |
| RGB count-failed scenes | 3 | 3 |

Failures overlap. Count failures stay in the 94-scene denominator. All six RGB
and all six Oracle successes have two objects: **0/23 three-object and 0/16
four-object scenes succeed**. No alignment violations were reported. Keep
violations remain zero, but this Full-only subset does not validate Partial
Keep generalization. Frozen Goal/Keep/Orientation outputs remain exactly
unchanged, and the selected checkpoint reload reproduces its complete Oracle,
RGB and diagnostic reports exactly.

The larger Teacher XY error is real, not hidden by lower gap error. Aggregate
Task improvement does not imply correct object-to-location assignment or
readiness for full Phase 1. Main, the existing Best registry and original
checkpoints remain untouched.

### Separate selection and placement diagnosis

On Oracle Validation, the freely selected sequence exactly matches Teacher in
**33/94 scenes (95/243 correct step positions)**. Exact Teacher order is stricter
than the evaluator's 5% size-tie tolerance; report it separately from Task order.
Of these 33 exact-sequence scenes, **27 still fail horizontal gap**.

| Oracle diagnostic | Task /94 | Counterfactual /23 | Gap failures |
|---|---:|---:|---:|
| Autonomous order and pose prefix | 6 | 1 | 80 |
| Teacher order, predicted pose prefix | 14 | 2 | 80 |
| Teacher order and preceding GT pose prefix | 7 | 1 | 87 |

Teacher-order intervention eliminates the 60 semantic order failures, yet
80 gap failures remain. This directly shows that selecting the right object
sequence alone is insufficient. GT prefix does not restore accurate spacing
either. It changes the state distribution of a decoder trained on predicted
prefixes; its lower success is not proof that autoregression itself is
incapable, nor a causal measurement of accumulated error alone.

Training mean selector CE falls from 0.52035 (updates1-250) to 0.02776
(2001-2250), and pose Huber from 3.42644 to 0.17574. These demonstrate training
objective improvement, not generalization: Validation sequence and gap results
remain poor. The observed bottlenecks are **sequence generalization and precise
spacing**, both present with Oracle inputs. This run does not identify which
further architecture/loss change would resolve them; no such search was run.

### Existing-success regression

Two of the prior four RGB successes remain; four new scenes succeed and two
previous successes fail:

- `expanded_7780ebdd60767f081d7bb067_0`: gap error 0.574 -> 6.536 mm;
  now fails gap only.
- `expanded_ad4fc129c886617c5718641b_4`: gap error 0.641 -> 6.841 mm;
  now fails gap and size order.

Thus the requested aggregate comparison passes, but this is **not** a
regression-free replacement for the protected Best.

### Actual learning trajectory

| Updates | Oracle Task | RGB Task | RGB CF | Exact sequence | Exact sequence + gap fail |
|---:|---:|---:|---:|---:|---:|
| 250 | 0 | 0 | 0 | 21 | 21 |
| 500 | 2 | 1 | 0 | 23 | 21 |
| 750 | 4 | 4 | 0 | 29 | 25 |
| 1000 | 5 | 1 | 0 | 29 | 25 |
| 1250 | 3 | 4 | 0 | 27 | 24 |
| 1500 | 2 | 2 | 0 | 26 | 25 |
| 1750 | 4 | 4 | 0 | 27 | 24 |
| 2000 | 6 | 5 | 1 | 30 | 25 |
| 2250 | 6 | 6 | 2 | 33 | 27 |

### Reproduction artifacts

All paths below are relative to the repository; retain the shared dependencies:

- `artifacts/neural_sequential_phase1_once/checkpoint-002250.pt`
  SHA-256 `5eeb7825c5fd503d0ad18dd6ad7eb478902a5926333f75c9d46c84edf76cfd11`.
- `artifacts/neural_sequential_phase1_once/shared/` and original pinned
  Perception/source-manifest dependencies.
- `result.json`: final gate, elapsed time, checkpoint replay and regressions.
- `evaluation-002250.json`: full coverage, per-scene targets/errors and both
  Teacher diagnostics; `analysis.json`: baseline comparison and strata.
- `history.jsonl`, `progress.json`, `learning-curves.png`, `train.log`:
  actual update losses, evaluation trajectory and output.
- `preflight.json`, `tests.log`, `preflight.log`, `execution.json`:
  validation results, immutable config/hash and training commit provenance.

The retained tmux session `knolling-neural-sequential` has exited successfully
(pane exit code0). `tmux attach -t knolling-neural-sequential` opens the retained
session; output was redirected to the artifact's `train.log`. This is not a
request to resume training. Disk free space after the run is approximately
27 GiB; no dataset or checkpoint cleanup was performed.
