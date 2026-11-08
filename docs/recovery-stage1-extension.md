# Stage 1 approved extension

The user approved 500 additional Arrangement updates from the initial run's
`checkpoint-000500-0022.pt`, with unchanged total limits of 1,000 updates and
600 cumulative driver seconds. Stage 2 remains gated by Stage 1's measured pass.

## Continuation contract

- Ordinary resume still rejects `budget_exhausted`. The separate `--extend-from`
  route requires an explicit receipt binding source checkpoint SHA256, original
  code/config/data, source update/time, new code/config and exact cumulative caps.
- Only the Arrangement update cap can change. Learning rate, precision, data,
  batch size, evaluation frequency and time cap remain identical.
- Model, AdamW moments, epoch/batch cursor, RNG and elapsed time are restored.
  Derived frozen-text cache is rebuilt in initial evaluation order before RNG
  restoration. The original rollback anchor and baseline are retained.
- Source files are read-only inputs; every continuation gets a new directory.
  An already extended exhausted checkpoint cannot silently receive another grant.
- At global update 700, compare the same full-suite position P95, Move accuracy
  and pure-axis success against final update 500. Stop if none improves; otherwise
  continue only to the existing total caps or an earlier Stage 1 pass. This is a
  fixed endpoint comparison, not selection of the best intermediate evaluation.
- The decision survives pause/resume and forces evaluation even if its update is
  not a regular evaluation boundary. A failed gate prevents subsequent stages.

## Validation

CPU/CUDA fixture tests compare interrupted-plus-extended and uninterrupted model,
optimizer, cursor and RNG states exactly. They also cover invalid receipts,
changed config, original resume rejection, source/anchor preservation, cumulative
time, persisted progress checks, early stopping and sequence gating. Real model
CUDA checks use forward/backward only, with no optimizer update.

Preflight: 106 Python files parsed; full CPU suite **213 passed / 22 skipped**
(46.36 s), including cached real-model checks. Explicit CUDA checks **2 passed**
(11.49 s): exact scalar continuation and real BF16 Arrangement / FP32 Perception
forward/backward. Real model weights/buffers stayed unchanged; peak allocation
was 641.8 / 2,836.7 MiB respectively.

## Result: Stage 1 failed; later stages not started

- Run: `recovery_stage1_extension_20261108T023824Z`.
- Executed commit: `6e1eb305c0208569fa033c455bee042b4366dcd5`.
- Completed exactly **500 additional updates**, ending at **1,000 total**.
  Driver cumulative time **335.21 s / 600 s**, including the original 172.09 s;
  this extension used 163.11 s. Model/cache setup is outside the driver's clock.
- Status `budget_exhausted`, sequence `stopped/update_budget`, no runtime error.
  Exit code 2 denotes unmet gate/budget stop. No retry or Stage 2/3 was launched.
- All metrics below use the same 96-record S0 memorization suite (271 objects,
  21 pure-axis pairs). These are not held-out generalization results.

| Metric | Update 500 | Update 700 | Update 1,000 | Stage 1 gate |
|---|---:|---:|---:|---:|
| Target position P95, mm | 140.84 | 139.89 | 110.43 | ≤1 |
| Target angle P95, degrees | 89.84 | 89.94 | 89.95 | ≤2 |
| Move accuracy | 76.75% | 80.81% | 85.24% | 100% |
| Pure-axis success | 0/21 | 1/21 | 0/21 | 21/21 |
| Full task success | 15/96 | — | 26/96 | Diagnostic |

At update 700, all three approved progress metrics improved over update 500, so
the persisted endpoint decision allowed continuation to 1,000. The later loss of
pure-axis success shows that this improvement was not stable. No gate threshold
was relaxed and no best intermediate checkpoint substituted for the final one.

## Remaining errors

Forward-only diagnosis of the final checkpoint (zero optimizer updates) found:

- Move/keep mistakes decreased from 63 to **40/271**: 23 moved objects incorrectly
  kept, 17 stationary objects incorrectly moved. The hard branch still produces
  large final pose errors when this classification is wrong.
- Candidate-position P95 on the 185 moved objects improved **75.07 → 47.35 mm**;
  after move/keep selection their deployed-position P95 is **134.22 mm**. Thus
  fixing Move classification alone cannot reach the 1 mm gate.
- Wrong horizontal/vertical orientation-bin choices decreased **55 → 38/185**.
  Candidate-angle median is 0.29°, but P95 remains 89.93°. Discrete orientation
  selection remains a major error despite accurate residuals on many objects.
- Axis auxiliary accuracy is **74.19%** (93 defined commands), versus 69.89%
  previously. Pure-axis paired behavior remains unsolved.
- Mean loss per 100-update block: **3.910, 3.733, 3.267, 2.974, 2.809**. Learning
  continued, but lower loss did not establish reliable instruction-dependent
  orientation, move decisions or sub-millimetre placement.

No further training is authorized by this extension. Before a new experiment,
review paired-command orientation/Move failures and their loss/gradient balance,
then determine whether changes or a separately approved budget are justified.
These observations do not establish a single causal defect or guarantee that
more updates alone will pass. No architecture, loss or data changes were made.

## Evidence and preservation

- Receipt, frozen config and console:
  `artifacts/recovery_launches/recovery_stage1_extension_20261108T023824Z/`.
- Run results, `evaluation-history.csv`, `extension-progress-check.json`,
  `arrangement-diagnosis.json`, `continuation-verification.json` and
  `preservation-check.json`:
  `artifacts/recovery_runs/recovery_stage1_extension_20261108T023824Z/`.
- Final checkpoint: `oracle_layout/checkpoint-001000-0022.pt` within that run.
- Actual continuation's initial checkpoint exactly matches the source model,
  optimizer, RNG, baseline, completed stages, epoch/batch/update cursor; original
  elapsed time is carried forward. This supplements the fixture equivalence tests.
- All **2,143 pre-existing artifact files** retain size and mtime; source final
  checkpoint and anchor SHA256 are unchanged. All new checkpoints are retained.
- Existing tmux session/window: `knolling-recovery-initial:stage1-extension`.
  Reconnect with `tmux attach -t knolling-recovery-initial`; training has ended and
  the window remains at a shell prompt. Existing notification watcher and sole
  read-only listener were preserved; no progress-notification worker was created.
