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
