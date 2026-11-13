# v4 integrated training implementation review

## Scope and execution status

Implemented against `artifacts/recovery_strategy/v4-integrated-training-plan.md`.
The v4 architecture, ordinary inference forward, Goal Encoder, Teacher and three
Pose losses are unchanged. This change is a **training procedure**, not a new
prediction model. No actual Knolling optimization run, Micro experiment or GPU
training was launched. The shipped config has `allow_training=false` and
`execution=review_only`; a new approval bound to the reviewed commit/config/data
is still required.

## Implementation map

| Requirement | Implementation |
|---|---|
| Coarse | `recovery/integrated.py`, `readout.PoseAdam`: all Pose parameters, AdamW 1e-4, existing weight decay and clipping, uniform record sampling |
| Global calibration | `readout.collect_readout`, `HeadFitter`: actual post-LayerNorm features, predicted frozen Goal, one common XY/angle readout for the complete TRAIN partition |
| Objective | XY: existing 10 mm SmoothL1; angle: existing 18-bin CE. Each scene has equal weight, then each valid object within that scene. Residual is untouched during calibration |
| Solver | Separate FP32 L-BFGS instances; strong_wolfe, history 20, max_iter=1 per outer call. Closure calls are counted separately; trial timeout restores last accepted leaves and history |
| Fine | Full-set maxima <=10 mm and <=2 degrees, only if Stage Gate still fails. LR 1e-6; shared/residual first moment reset once, second moment and step preserved; XY/angle optimizer state starts fresh |
| Final calibration | Feature cache and both L-BFGS histories rebuilt after fine; one final XY then angle fit, no repeated rounds |
| Inference | Original fused 38-row Linear and state-dict keys retained; no solver, Teacher or per-scene head in RGB+language inference |
| Acceptance | Existing Micro / 76-scene / 35-pair Gates unchanged. New diagnostics retain identities of passing scenes/pairs and count Keep violations; Fine also retains previously correct orientations |
| Transfer | `provenance.py`: checkpoint SHA, original config/commit, explicit allowed config changes, model/loss/Teacher/evaluator semantic hashes |
| Future families | `family_manifest.py`: separate pinned manifest, loader and descriptive evaluation API; existing verification loader/Gates remain strict |

`PoseAdam` uses six optimizer-only leaves for XY/logit/residual weight and bias.
This permits independent Adam step counters without changing the model Linear.
An ephemeral forward hook computes the same fused affine expression. After each
accepted Adam step, values are copied into the original model parameters.
Calibration optimizes cloned selected rows and publishes only accepted rows.
Gradient masking of a shared Adam tensor is not used.

The feature cache binds producing model weights, dataset fingerprint, labels,
and sample/object/query mapping. The output Linear is excluded from the producer
hash because it cannot affect preceding features. Feature caching rejects held-out
records, duplicate/incomplete scenes and duplicate object identities.

## State machine and budgets

| Phase | Micro iterations | Full-76 iterations |
|---|---:|---:|
| Coarse | 80 | 200 |
| XY calibration | 20 | 50 |
| Angle calibration | 20 | 50 |
| Fine | 40 | 100 |
| Final XY | 20 | 50 |
| Final angle | 20 | 50 |
| Total cap | 200 / 120 s | 500 / 600 s |

Each calibration phase also has a 10-second cap. These are proposed config caps,
not newly authorized experiments. Unused iterations are not reassigned.

- Coarse reaches calibration on its cap even if 10 mm / 2 degrees is not met;
  meeting that full-set threshold early also switches to calibration.
- XY max <=1 mm and angle minimum true-bin probability >=0.9 independently stop
  the respective solver. Solver convergence/caps also end it. These diagnostic
  criteria **never replace** the actual Stage Gate.
- If the Stage Gate passes at an evaluation, execution stops immediately.
- If B cannot enter Fine, status is `fine_ineligible`; no lower-LR retry occurs.
- Regressions after Coarse restore the phase-entry model and terminate as
  `rolled_back`. Timeout performs final evaluation and regression checks.
- Final evaluation and durable checkpoint publication are mandatory even after
  the optimizer deadline; shutdown overhead can exceed that deadline, with no
  extra optimizer update.
- Each independent Micro starts from the same passed Grounding file. Oracle
  starts from that Grounding, not a Micro-trained model, and requires all three
  passes with the **new integrated recipe**. Old Head-only logs are not evidence.

## Persistence, resume and provenance

Full checkpoints retain model, split Adam leaves/moments, phase/subsolver state,
iterations, closure count, cumulative time, RNG, data/code/config contracts,
Grounding SHA, phase anchor hash, feature cache hash and phase outcomes.
Accepted calibration steps first publish a small atomic `readout-*.pt` journal
with learned weight/bias, solver state and a hash-bound immutable full checkpoint.
This publication precedes observer logs, evaluations and ancillary reports.
Observer failures are recorded; they do not discard learned weights.
Full checkpoints and JSON results use exclusive atomic publication and fsync.
Existing output folders/files cannot be overwritten.

Resume accepts only interrupted `running`/`paused` snapshots with identical
commit/config/data. A `terminal.pt` marker rejects reopening an older snapshot
from a finished run. Exhausted/failed runs have no implicit extension path.
Keep the output directory, phase anchors, feature caches and journal base files
available: journals are not standalone inference exports. The terminal full
checkpoint is standalone for model weights; its resume dependencies remain
explicitly hash-bound.

Cross-commit evidence transfer is separate from resume. A review receipt must
pin `sha256`, `commit`, `config_sha256`, `contracts` and
`allowed_config_changes` in `grounding_transfer` (or a matching entry of
`micro_evidence_transfer`). Allowed differences are only execution/approval
metadata and the precision/integrated policy fields. Budgets, model/data and Loss
cannot silently change. Contracts are AST hashes of the enumerated model,
loss, Teacher and evaluator sources; only explicitly named reporting fields
(`micro_predictions`, `integrated_evidence`, `breakdown`) and the breakdown import
are excluded. The real successful Grounding source commit `7ac097a` has matching
contracts. No transfer grant or training approval was issued by this implementation.

## Future-family support boundaries

No new dataset was generated. The separate `FamilyDataset` API expects a pinned
`recovery_family_pilot_v1` manifest containing Teacher provenance, annotation path
and hash, family partition/origin/source/RGB hashes and exact partition coverage.
It requires legacy25 train + new40 train/new10 validation/new10 test, all modes
and counts 2/3/4, all six pair factors, and at least five validation/test families
per factor. It rejects split-crossing pairs, duplicate observable inputs and
source/RGB duplicates (source hashing ignores IDs/order), replays the Teacher,
and verifies RGB/annotation hashes. Pure-pair validation reuses the existing
verified validator.

`evaluate_family_pilot` reports per-family/mode/object-count/factor results and a
separate >=95% success, >=90% mode/factor, zero collision/Keep violation criterion.
It cannot authorize a production StageEngine transition. The current CLI still
uses the existing 76-record verification bundle. Generating a genuine new pilot,
training it, validation-based model selection and one-time test orchestration
remain future separately approved work. The new APIs were checked with metadata
fixtures and exact-output evaluation, not a newly rendered 85-family dataset.

## Verification evidence

See the test results recorded below. Synthetic fixture tests perform small
optimizer steps solely to validate state transitions. The actual cached-model
CUDA test performs **zero optimizer steps**, checks unchanged model weights,
frozen Goal/Perception/MiniLM, finite shared/head gradients, cached-head versus
ordinary-forward equality, and the RGB inference path. RTX 4090 peak allocation
for that test was 982 MiB.

Required review targets:

1. Split Adam mapping and fine reset: do shared/residual second moments and step
   survive while both calibrated heads receive fresh states?
2. Accepted-step journals: are restored fused weights, old Adam history, solver
   state, cache and RNG consistent before/after every phase transition?
3. Phase-local retention versus Gate: no probability-only pass, aggregate-only
   regression check, skipped final timeout evaluation or automatic extension.
4. Scene weighting and complete TRAIN-only mapping; cache invalidation after
   Fine; unchanged predicted-Goal and RGB inference paths.
5. Evidence-transfer scope and the distinction between verification and future
   generalization. Review any future receipt against its exact source commit.

Remaining empirical risks: shared-feature linear separability over all76 is
unproven; L-BFGS may hit its ten-second cap; residual-angle errors can block Fine;
strict regression guards may stop a useful but non-monotone candidate. Passing
synthetic tests is not proof that H/V Micro or full Oracle will pass. The next
actual experiment remains a separately approved integrated H/V Micro; no automatic
follow-on training is enabled.

### Completed checks (2026-11-08 UTC)

- `knolling` Python `/venv/knolling/bin/python`, PyTorch 2.9.1+cu128,
  CUDA 12.8, NVIDIA GeForce RTX 4090.
- Full CPU neural regression suite: **213 passed, 34 skipped**, 87.02 s.
  The skipped CUDA/optional cases were not counted as passes.
- Focused CPU/CUDA suite, including newly added line-search deadline and corrupted
  cache guards: **83 passed**, 64.56 s. Covered readout, integrated lifecycle,
  v4 Goal/permutation/padding/gradient, runtime, precision, driver and evaluation.
- All six phases reproduced exact model/optimizer/RNG states after pause/resume
  on CPU and CUDA fixtures; journal-only resume also matched uninterrupted output.
- Real-model integrated CUDA test: FP32 forward/backward and readout equivalence,
  **0 optimizer steps**, peak 982 MiB. Existing v4 FP32/BF16 forward/backward
  regressions also passed; perception check peak 2928.92 MiB.
- `compileall` and `git diff --check`: passed.
- Existing **2,388 artifact files** retained their sizes and nanosecond modification
  times. Existing dataset loaders revalidated the pinned annotation/Teacher/RGB
  hashes. No existing dataset, checkpoint or result was overwritten.
- `goal_joint_layout.py`, `goal_encoder.py`, `joint_layout.py`, `stages.py` and
  `teacher_v3.py` have no changes relative to pre-implementation `24744f7`.

Reproduction (inside the `knolling` environment):

```sh
python -m pytest -q tests/neural
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
CUBLAS_WORKSPACE_CONFIG=:4096:8 RUN_RECOVERY_CUDA_TESTS=1 \
python -m pytest -q -s \
  tests/neural/test_recovery_readout.py \
  tests/neural/test_recovery_integrated.py \
  tests/neural/test_recovery_goal_v4.py \
  tests/neural/test_recovery_v4_runtime.py \
  tests/neural/test_recovery_pose_precision.py \
  tests/neural/test_recovery_driver_v4.py \
  tests/neural/test_recovery_evaluation_v4.py
```

### P1 correction: interrupted Head candidate evaluation

Independent review of `7aff254` found that a journal saved before evaluation could
resume with an extra solver update. This could skip the candidate's PASS or
regression decision and lose the solver's pending phase-exit decision.

The driver now persists `pending_head.phase_done` **before publishing the
accepted-step journal**. Both uninterrupted and resumed execution resolve that
candidate through the same full evaluation / Gate / regression / phase-advance
function. Resume resolves it before any new optimizer step, including when the
time budget or stop flag is already set. A resume checkpoint retains the pending
state, so a second crash does not lose the decision. Stage completion, solver
convergence, diagnostic criterion and iteration-cap exits are preserved.

Existing journal files are never rewritten. A journal lacking the new field can
recover its pending decision from the saved diagnostics and phase counter;
existing commit/config/data validation and terminal-run restrictions still apply.
This is not permission to resume an old commit under different code.

`test_recovery_integrated_resume.py` simulates a process death after the journal
is durable, bypassing graceful-pause cleanup. It covers all four calibration
phases with PASS, regression, solver convergence and iteration-cap outcomes on
CPU/CUDA. It compares exact update/cursor counts, final model weights, Adam and
L-BFGS states, RNG, reset/closure counters and final metrics against uninterrupted
execution. It also covers a second crash during resume, old-field journal loading
and an expired budget. Optimizer updates are confined to synthetic tensor
fixtures; the existing cached real-model CUDA test performs forward/backward only.

Correction validation in `knolling`: CPU lifecycle suite **40 passed / 25 CUDA
skipped**; focused CPU/CUDA suite **87 passed** (55.05 s), including readout and
precision regressions. `compileall` and `git diff --check` passed. All **2,390**
pre-existing artifact files retained their size and nanosecond modification time.
No actual model training was launched, and the review-only config remains disabled.
