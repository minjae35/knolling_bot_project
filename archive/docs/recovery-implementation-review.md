# Neural knolling recovery — review after the fixes

**Status: code and CPU/CUDA checks are done. No real model training and no performance experiment has been run. Waiting for a work review.**

- Branch: `feat/neural-recovery`. Baseline before the fixes: `9b141c3`.
- This note updates the earlier implementation report. That report is in Git at `9b141c3`.
- Run config: [`recovery_execution.json`](../configs/recovery_execution.json). The default is `review_only` and `allow_training=false`.
- Every Python check used the conda environment `knolling` (`/venv/knolling`). No extra packages were installed.

## 1. Review findings and the actual fixes

| Finding / confirmed cause | Fix | Evidence checked |
|---|---|---|
| Arrangement training matched all queries with Hungarian matching, while inference selected foreground ≥0.5 | Shared `select_queries()` selects the same queries first, then matches only the source answers inside that set. A count, capacity, or 10 mm match failure raises `SourceSelectionError` | A regression fixture that split a confidence boundary into train `[0,2]` and inference `[0,1]`; selected-query identity and a CUDA train/inference output comparison |
| The evaluation output and the gate input used different schemas, so a real measurement could not pass; one target was reported as if it were three stages | `recovery_metrics_v2` is wired to the real evaluator. Oracle, perception, and predicted paths are separate, and every expected sample ID must be present for the run to be complete | Four gates checked on real evaluator output; missing evaluations, ignored axis commands, and duplicate-detection failures were confirmed. A fixture pass is not model performance |
| Stage functions existed, but there was no run, stop, or checkpoint operation | A driver that runs one stage per call, requires the previous gate, limits updates and time, and implements checkpoint, resume, and rollback | On a scalar fixture, continued and resumed runs match weights, Adam state, and cursor; skipping a stage is refused; nonfinite, selection, and regression rollbacks |
| Counterfactuals only had an instruction builder; no new teacher or loader was connected | 64 checked target records were generated and joined to the original 32 through `RecoverySuite`. Teacher replay, hash, family/split, and English meaning are checked | All 32 pairs distinguish both command directions and pass geometry; 64 teacher targets reproduce; the regenerated bundle SHA matches |

Code: [`selection.py`](../src/knolling_model/recovery/selection.py), [`evaluation.py`](../src/knolling_model/recovery/evaluation.py), [`stages.py`](../src/knolling_model/recovery/stages.py), [`driver.py`](../src/knolling_model/recovery/driver.py), [`counterfactuals.py`](../src/knolling_model/recovery/counterfactuals.py).

## 2. Architecture that stayed, and the ground-truth boundary

```text
BEV RGB → RT-DETR-R18 foreground / physical heads
          → select foreground queries at inference
          → xy, size, periodic angle, color/shape probabilities, query identity
English → frozen MiniLM → trainable projection
          → object/text Transformer → autoregressive target pose + move/keep
```

- Inference receives only RGB and tokenized English. It is not given a ground-truth count, attributes, a target prefix, a command parser, the teacher, or a rule-based layout planner.
- Only the oracle stage receives the ground-truth source, and that is an explicit diagnostic. In predicted and joint mode, ground truth is matched to loss labels **after** query selection. Missing detections are not filled in from ground truth.
- Joint training keeps the layout-to-perception gradient through continuous source attributes. This note does not claim that the foreground threshold, the move decision, or the discrete orientation-bin choice is itself differentiable.
- The existing denoising split and mask, 1-class foreground, π-periodic bin and residual, learned keep, and the separate unique-source detection replay were kept.
- This fix keeps the legacy evaluator's default three-stage behavior and adds only an optional `stage_count`. A recovery target is one pose. A perception-only report marks the layout result as not measured.

## 3. Real evaluation and the stage pass criteria

Every gate requires the correct schema and stage, free rollout, and an exact match to the pinned full sample-ID set. A sample that drops an object or exceeds capacity remains a failure.

| Stage | What is trained / data | Pass criterion | Run cap |
|---|---|---|---|
| `oracle_layout` | ground-truth source + English, layout | target P95 ≤1 mm / 2°, move 100%, pure-axis 100% | 500 updates / 600 s |
| `perception` | RGB, 25 unique sources | count and coverage 100%, FP/FN 0, source position and size P95 ≤1 mm, angle ≤2°, color/shape 100% | 1,000 / 2,700 s |
| `predicted_layout` | predicted source + English, perception frozen | source criteria plus the legacy S0 gate plus full task and pure-axis 100%, no per-mode regression | 1,000 / 1,800 s |
| `joint` | layout and the detector's last decoder and heads, plus a separate detection replay | the criteria above plus no overall regression against the stage-start reference | 300 / 900 s |

- FP and FN are counted by an independent match: source center distance ≤10 mm and footprint IoU ≥0.5. A raw count difference does not cancel them.
- Pure-axis, on a checked pair that changes only the axis of the same RGB, requires **both own commands to succeed and both opposite commands to fail**. A missing pair, or a pair with only one side scored, does not pass.
- The denominator for color, shape, and move accuracy is every ground-truth object. Undetected objects are not removed from the denominator.
- The overall regression comparison, on the same sample set, includes count, coverage, FP/FN, source and target P95, task, axis, move, color, shape, and per-mode task success.
- The oracle target P95 criterion is not the maximum error over every object. Predicted and joint stages additionally require full semantic and geometry task success.
- **There is not yet a result in which a real model passed these gates.**

## 4. Dataset and teacher checks, and their limits

New bundle: `artifacts/recovery_v2_controls/counterfactuals.json`.

- **64 records / 32 pairs:** 21 pure-axis pairs, 5 region pairs, 6 selection pairs.
- The suite that also includes the original 32 S0 scenes has **96** records and shares the original **25 source families**. RGB was not re-rendered.
- The existing split is inherited. This suite is for S0 repair and command discrimination inside train. **It is not held-out generalization data.**
- The loader checks the parent manifest, source, source RGB SHA, family and split, pair structure, agreement between the English sentence and the structured command, and geometry and semantic validity. On the default load it reruns the new teacher and compares the target.
- Cases where the old teacher and the new teacher attach **different targets to the same source, RGB, and sentence** were found, excluded at generation, and rejected by the loader. The old labels are left as they are.
- The new `continuous_envelope_v0_review` teacher is offline only and does not replace the existing default teacher. It is applied only to the new contrast targets.
- Continuous position projection, locked-object preservation, no-op/full/partial validity, a 10 μm local perturbation, and a distance check of 20 obstacle-union fixtures against a lattice reference all passed.
- Discontinuities remain at template switches and at ties on both sides of an obstacle. Because the envelope is conservative, a feasible problem can still be rejected as a generation failure. Large-scale stability and generalization were not shown.

Bundle SHA-256:
`2110de927ab86fc3b0b6e2a7f8a768a61d71e7ef4da23d13136f58217facccac`

Suite fingerprint:
`e515a4694af4a57bc803b0a40e86665e96f596bd30aa8f01de834b485ed7db2b`

The bundle is a local artifact and is not in Git. On a fresh checkout that has the same parent dataset, regenerate it with the command below. An output folder that already exists is not overwritten.

```bash
conda run --no-capture-output -n knolling python -m knolling_model.recovery.counterfactuals \
  --output artifacts/recovery_v2_controls --pairs 32
```

## 5. Driver operating contract — not approved to run

- CLI: `python -m knolling_model.recovery.driver`. It must be used in the `knolling` environment.
- `--stage`, a new `--output`, and `--approval` are required. The next stage uses `--parent-checkpoint`. Resuming the same stage uses `--resume`. The two options cannot be used together.
- The default config refuses to run before creating the model, CUDA state, or output. A later real run needs a separately reviewed config with `execution=approved_training`, `allow_training=true`, and a review receipt.
- Receipt fields: `approved`, `reviewer`, the exact `commit`, `config_sha256` (`schema.digest(config)`), `dataset_fingerprint`, and the allowed `stages`. CUDA training also needs `gpu_training=true`. This is a review record that blocks an accidental run. It is not a signed security certificate.
- The checkout must be clean and the code, config, and data must agree. The config and the receipt are also stored in the checkpoint. This work did not create a real training-approval receipt.
- A checkpoint stores the model, optimizer, Python/NumPy/Torch/CUDA RNG, update/epoch/batch cursor, accumulated time, baseline and completed gates, and the stage-entry anchor hash. It is written so that an existing file is not atomically overwritten.
- SIGINT and SIGTERM pause at a step boundary. A batch interrupted during forward or backward, before the update, restores the RNG and is recomputed on resume. A hard kill resumes from the last durable checkpoint.
- Every run and every resume requires a new output directory. Only `running` and `paused` can be resumed. `budget_exhausted` and `rolled_back` are not resumed automatically.
- A nonfinite loss, gradient, or parameter, a selection failure, or a joint regression rolls back to that stage's start anchor and leaves an error checkpoint. Optimizer moments in a rollback checkpoint are a record of the rejected path and are not used to resume.
- The next stage is not entered without a checkpoint that passed the previous gate. The default `stop_on_initial_pass=true` ends a stage that already passes, with no update.
- The time limit is a cooperative limit on optimizer updates. It does not forcibly cut a kernel, evaluation, or checkpoint write that is already running, so the process can exit after the cap.
- If time expires during forward or backward, the gradient of the unfinished update is cleared, the RNG is restored, and control moves to the shared final evaluation. A joint regression rolls back to the stage-start anchor. A final-evaluation error is also handled as a rollback. With no regression, the run ends as `passed` or as `budget_exhausted` including the evaluation, according to the final gate.
- The default path is free rollout. Source-noise and teacher-forcing helper APIs remain, but no measured noise policy exists, so the driver does not turn them on.

## 6. Verification results

The base record below is at `9232b4c`. Checks after the timeout fix are in section 6.1.

CPU reproduction:

```bash
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
  conda run --no-capture-output -n knolling python scripts/verify_recovery_cpu.py --cached-model
```

CUDA reproduction — forward and backward only, with no optimizer creation or step:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RUN_RECOVERY_CUDA_TESTS=1 \
  conda run --no-capture-output -n knolling python -m pytest tests/neural/test_recovery_cuda.py -q -s
```

- AST: **100 Python files**; `git diff --check` passed.
- Full CPU pytest: **187 passed, 12 skipped**. This includes the real cached R18/MiniLM CPU shape and gradient check.
- Skips: 2 CUDA tests (the recovery CUDA check is run separately below), 2 opt-in legacy pretrained tests, and 8 rendering, migration, and full-pilot checks. Blender was not regenerated.
- Driver unit tests: **6**. Only a scalar fixture was optimized. That is not training the real knolling model.
- CUDA: **1 passed**, RTX 4090, Torch `2.9.1+cu128`, CUDA `12.8`. Finite forward and backward were checked for perception/denoising, oracle layout, predicted layout, and joint, plus the layout-to-perception gradient and the joint detection-replay backward.
- The predicted and joint CUDA paths used a synthetic selection fixture that only controlled the confidence of an untrained detector. The untrained model itself was also confirmed to raise `SourceSelectionError`. This check does not score accuracy.
- GPU peak allocated **1,146.4 MiB**, reserved **1,240.0 MiB**. That is a batch-1 measurement. It is not an estimate of optimizer state or of the real training peak.
- Real model **optimizer updates: 0**. A full state-dict SHA comparison before and after the check confirmed that weights and buffers did not change. No GPU compute process remained at exit.
- Path, size, and mtime comparison of the existing **2,105** files under `artifacts/`: 0 changed or deleted. The only additions were the new bundle and two SHA files. This is not a re-hash of every byte of the existing checkpoints.

### 6.1 Checks after the timeout-exit fix

- Before the fix, 8 new CPU regression tests all failed. After the fix, all of them passed.
- For a timeout during forward and during backward, the checks confirmed one final evaluation, no extra optimizer update, joint-regression rollback, budget exit without regression, a passing gate, and rollback when the final evaluation errors. The checks read the saved checkpoint's metrics, cursor, completed stage, and restored weights, not only the return value.
- Full CPU: **195 passed, 20 skipped**. Existing pause, resume, and stage-transition checks are included. Eight new CUDA timeout cases skipped on CPU all passed in a separate GPU run.
- Separate CUDA: **9 passed** — 8 scalar driver cases and 1 real R18/MiniLM forward/backward check. The real knolling model had 0 optimizer updates, and weights and buffers were unchanged.
- The **2,107** artifacts that existed before this fix kept their path, size, and mtime. Existing data and checkpoints were not overwritten, and real GPU training was not started.

CUDA reproduction command:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RUN_RECOVERY_CUDA_TESTS=1 \
  conda run --no-capture-output -n knolling python -m pytest \
  tests/neural/test_recovery_driver.py::test_joint_timeout_final_evaluation_and_rollback \
  tests/neural/test_recovery_cuda.py -k 'cuda or real_r18' -q -s
```

## 7. Work-review priorities

1. **Query and ground-truth boundary:** whether the selection policy is the same in train and inference, whether source matching is used only to join labels, and whether the strict abort matches the intended stage 3 and 4 operation.
2. **Evaluation and gates:** the complete sample set, the split between oracle and predicted paths, physical FP/FN, the two-direction pure-axis condition, the source-only report, and the regression denominator and tolerance.
3. **Execution:** the checkpoint contract, stage order, data-cursor and RNG resume, the accumulated time limit, stage-entry rollback, and states that must not resume. After review, fix the config and the caps that will be approved.
4. **Teacher and data:** provenance and reproduction of the new labels, blocking collisions with old labels, the conservative envelope, and boundary discontinuities. Remember that this rests on the existing 25 families, and keep it separate from generalization data.
5. **Remaining design risks:** autoregressive stability when detector query order changes, the gradient limit of hard move and bin choices, the scale between losses, and whether the high source gate is achievable. These tests do not show convergence, accuracy, or a real knolling success rate.

This fix produces **an implementation that can be reviewed again**. Whether real GPU training starts is decided after the work review.
