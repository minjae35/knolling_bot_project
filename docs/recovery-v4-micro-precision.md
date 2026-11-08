# v4 Micro Pose precision phase

This opt-in driver change addresses excessive AdamW steps near the translation
target without changing architecture, Goal Encoder, loss, data, or Micro Gate.

- Config: `configs/recovery_v4_micro_precision.json`. Only the added
  `micro_precision` policy differs from the Grounding parent's config. The
  inherited `budget_status` text describes the older Grounding approval; the
  current exact-code/config/data receipt is the execution authority.
- Scope: v4 `micro_pose` only. Evaluate after every completed update. When the
  complete oracle evaluation reports maximum position error <=10mm AND maximum
  angle error <=2 degrees, switch every Pose parameter group from 1e-4 to 1e-6.
- At that transition only, zero AdamW `exp_avg` in place. Keep `exp_avg_sq`,
  optimizer step, weights, betas, epsilon, weight decay, and clipping unchanged.
- Save `pose_precision.phase` and `transition_update` with the optimizer. Resume
  validates phase/LR consistency and rejects missing phase state. It never
  repeats the reset. Paused/running-only resume and cumulative budgets remain.
- Per-update JSONL contains pre-update loss/gradient norm/LR and post-update
  evaluation with per-object poses and errors. These provide centroid movement
  and sign-reversal diagnostics; they do not enter the loss or inference path.
- Final evaluation, regression rollback, immediate gate stop, and immutable
  checkpoint publication remain in the common driver path.

## Grounding transfer after the code change

Normal checkpoint loading still requires identical commit/config/data. A fresh
Micro run can import a passed production Grounding checkpoint only when its
exact SHA256, source commit, and source config digest are explicitly listed in
the current receipt's `grounding_transfer`. The source is validated under its
original contract, data must match, and the new config may differ only by the
precision policy. Only model weights are inherited: Micro starts at update 0
with a new Pose optimizer. The original file is never rewritten or relabeled.
This path cannot resume a failed Micro checkpoint or extend its budget.

The approved rerun is translation only, once, <=200 updates / <=120 seconds,
starting from the successful 450-update Grounding parent. The receipt restricts
the case to translation; no coordinator or subsequent stage is launched.

## Validation

`test_recovery_pose_precision.py` covers inclusive thresholds, nonfinite/incomplete
metrics, all-Pose group ownership, preservation of weights/second moment/step,
one-time reset, CPU/CUDA bitwise-equivalent interrupted execution before/at/after
transition, immediate gate termination, update budget exhaustion, resume denial,
case authorization, and pinned Grounding transfer rejection paths.
Existing driver lifecycle and v4 forward/backward/evaluation tests are also run.

The new LR is a hypothesis to validate with one bounded run, not a claim of a
passed Micro Gate. Keep-all, axis, and Oracle Layout remain unapproved here.

## Completed translation rerun — 2026-11-08 UTC

Implementation commit: `fa60fdb66d299c73eb6b06e1b22a19fa822c2437`.
One translation run started from the original passed 450-update Grounding
checkpoint, never from the failed/budget-exhausted Micro checkpoint.

**PASS at update 63, 10.354 seconds**, then stopped immediately. The original
200-update / 120-second caps were not extended. Keep, axis, and Oracle Layout
were not started.

| Measurement | Original fixed LR | Precision rerun |
|---|---:|---:|
| Updates | 200 | 63 |
| Driver elapsed seconds | 21.030 | 10.354 |
| Max target position error, mm | 4.195610 | 0.519057 |
| Max angle error, degrees | 0.111701 | 0.843713 |
| Centroid error magnitude, mm | 4.150887 | 0.117666 |
| Max error after removing centroid, mm | 0.054298 | 0.490677 |
| Knolling Validity | PASS | PASS |
| Micro Gate | FAIL | PASS |
| Max position error at matched update 50, mm | 25.0721 | 1.69696 |

At update 34, max position error was 8.720268mm and angle error 0.823721 degrees.
The LR changed once; update 35 was the first fine step, moving the centroid
0.110339mm. There were 29 fine updates: mean centroid movement 0.435130mm,
maximum 0.699835mm. Small oscillations remained (three X-error sign crossings,
two Y-error sign crossings, five increases in max position error between fine
evaluations). This was not monotonic convergence or proof of complete stability.

The first 25 losses and checkpoints 0/25 match the original run exactly, including
bitwise model weights. Improvement at matched update 50 supports a real optimizer
effect, beyond the changed evaluation frequency. Original evaluation ran every
25 updates; the new run evaluates every update, so the original's earliest possible
passing update is unknown. LR reduction and moment reset were applied together;
this single experiment cannot isolate their individual effects.

The early-stopped run has less accurate relative layout/angles than the 200-update
original, but both are within the unchanged Gate. Its last-25 mean total loss
(1.67123) is higher than the original's (0.63076): termination uses actual pose
and validity, not total loss. This is one two-object memorization case, not evidence
of generalization, full Stage 1 success, or approval for subsequent experiments.

### Verification and preserved evidence

- 69 tests passed in `knolling`: driver/precision CPU and CUDA (44), actual v4
  FP32/BF16 forward/backward and goal/evaluation checks (23), ownership/timeout
  exception paths (2). Scalar fixture tests perform optimizer steps; actual-model
  validation before the authorized run performs forward/backward only.
- Actual Micro GPU forward/backward peak allocation was about 611 MiB. The bounded
  experiment retained FP32 to match the original precision experiment.
- Parent anchor equals original Grounding weights exactly. Only 36 Pose state
  tensors changed; Goal Encoder, MiniLM, and Perception stayed bitwise unchanged.
- Final checkpoint has phase `fine`, transition update 34, optimizer step 63,
  and LR 1e-6. Resume equivalence is tested before/at/after transition on CPU/CUDA.
- All 2,320 pre-existing artifact files retained size/mtime. The original failed
  checkpoint SHA256 and Grounding parent SHA256 were also reverified.
- No model/loss/data/goal module changes, no extra training, and no later cases.

Run directory:
`artifacts/recovery_runs/recovery_v4_micro_precision_20261108T044523Z/`.
Per-update evidence is in `translation/history.jsonl`; `analysis.json` contains
per-update centroid movement, error vectors, oscillation counts, and preservation
checks. Launch/config/receipt/audit scripts are preserved under the same name in
`artifacts/recovery_launches/`.

Final checkpoint: `translation/checkpoint-000063-0004.pt`.
SHA256: `6f55e161dd804908de77048f545cdd88c09fbd57c6d4e01be95e50aad5334d00`.

The completed terminal is retained in tmux session `knolling-v4-micro-precision`:
`tmux attach -t knolling-v4-micro-precision`.
