# XY Feature diagnostic and exact alternating evaluation

## Confirmed failure causes

The single alternating trial `recovery_v4_alternating_20261108T081816Z`
stopped before any Feature update. Its 50 saved Orientation journals show CE
falling from 1.00627160 (iteration 1) to 0.04561909 (iteration 50), but minimum
true-bin probability was only 0.00496415 versus the required 0.90, and the
solver's numerical convergence flag remained false. A low mean CE does not
imply every object is confident. This was the `inner_not_converged` iteration
cap, not a time limit, NaN or missing gradient. No further CE fitting was run.

The reload discrepancy was independently reproduced using the unchanged real
checkpoint and all 76 scenes. Fully frozen evaluation differed from Driver
evaluation by 0.0000178543 mm in SVD RMS and at most 0.0001228771 mm in per-object
Keep position error. Restoring the Driver's shared-feature `requires_grad`
flags made every diagnostic exactly equal to the saved record. Separate probes
held the text-cache tensor hash fixed and changed stage/train/eval modes:
shared-feature flags reproduced the record; fully frozen flags did not.
The model state itself was bitwise identical. This isolates a floating-point
execution-path difference associated with parameter flags, not lost weights.

`alternating_evaluation` now normalizes parameter flags and eval mode under
`no_grad`, restoring all original flags and module modes on exit (also on
exceptions). It does not clear/change gradients or take an optimizer step.
Driver evaluation and `trial_measurements` use this context. Standalone full
reload evaluation must use `evaluate_alternating` instead of calling the raw
`evaluate_v4` with a different freeze configuration. No tolerance or Gate was
relaxed. The old one-off experiment script and all historical artifacts remain
unchanged.

The corrected real-checkpoint CUDA reload exactly reproduces all diagnostics,
Teacher Fidelity, Validity, contrasts, Grounding and integrated evidence; model
weights remain unchanged and real-model optimizer steps are **zero**.

## Separate diagnostic path

`configs/recovery_v4_xy_diagnostic_review.json` adds only
`alternating_pose.diagnostic = "xy_feature_only_v1"` to the existing policy.
It is review-only with `allow_training=false`. Source SHA, dataset, optimizer,
LR, clipping, architecture and all three Pose loss terms are unchanged.

The diagnostic performs:

1. Record the same full-set baseline and SVD RMS.
2. Calibrate XY with the existing SVD-Huber solver. Keep Orientation and Residual
   rows unchanged; never enter the CE solver, initially or after Feature updates.
3. Run actual-forward orientation/Keep/task non-regression checks and the
   existing 5% RMS + H/V improvement diagnostic Gate.
4. If unfinished and retained, freeze all Head rows and update only shared
   features with the unchanged Pose loss over all 76 records.
5. Invalidate/recompute features, recalibrate XY, then check again.

The original alternating policy still calibrates both Heads, and the formal
Arrangement Gate/evaluator is unchanged. New checkpoints have
`purpose=diagnostic`; their PASS is not a formal Arrangement PASS. Config hashes
prevent reinterpreting interrupted old trials as this new policy. Source
approval, Journal decisions before updates, anchor/optimizer/RNG rollback,
10-update/120-second caps and terminal-resume rejection remain enforced.

## Verification and remaining limits

New synthetic CPU/CUDA tests exercise real SVD/feature code and interrupt after
XY journals, Feature updates and phase transitions. They compare uninterrupted
and resumed weights, optimizer, RNG, counters and metrics exactly, verify fixed
Orientation/Residual/Goal weights, and cover regression/deadline rollback,
FP32 stall continuation, the update cap and diagnostic purpose enforcement.
Tests isolate lifecycle decisions with controlled metrics; they do not assert
real-world learning success. Separate real-forward reload checks use the
historical 76-scene dataset without updates.

A frozen Orientation Head can still change predictions when features change.
The unchanged non-regression checks may correctly stop this diagnostic, even
before its first update if initial XY calibration regresses Keep/task behavior.
The real-data XY improvement and 120-second fit remain unverified: no new real
training, Micro rerun, solver fitting on real data or budget extension occurred.

Local read-only evidence and test logs:
`artifacts/recovery_diagnostics/xy_diagnostic_revision/`.

Validation: **220 CPU/CUDA tests passed** across alternating, the new XY
path, integrated lifecycle/resume/efficiency, SVD, Coarse extension, readout,
storage and v4 Driver. The real-model CUDA readout check uses Forward/Backward
only, with no optimizer step. Original 2,611 files retain size, mtime, inode and
link count; the pinned source and annotation SHA256 values are unchanged.
