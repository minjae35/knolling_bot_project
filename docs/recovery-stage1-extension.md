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
was 641.8 / 2,836.7 MiB respectively. Execution results follow after the run.
