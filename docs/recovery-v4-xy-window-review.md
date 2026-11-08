# Provisional XY diagnostic window

## Confirmed regression

CPU replay of the immutable 600-update source and the previous XY journal
(`recovery_v4_xy_once_20261108T083816Z`, `readout-000000-0026.pt`) identifies
`vertical_gap` as the sole failure of scene `v3_a3d6fc3fc6211e89b6d5aa60`.
Two differently colored row bands require a 40 mm gap with 2 mm tolerance.
The source predicts 41.982643 mm (1.982643 mm error); XY calibration predicts
31.971327 mm (8.028673 mm error). Geometry otherwise remains valid. Source,
annotations and previous run artifacts were read only.

## Scope

`configs/recovery_v4_xy_window_review.json` selects the separate
`xy_feature_window_v1` diagnostic. Both older alternating policies and the
formal Arrangement Gate are unchanged. The new policy retains source SHA,
all 76 records, SVD-Huber XY calibration, fixed final Heads during feature
updates, all three Pose loss terms, optimizer/hyperparameters and the original
10-update/120-second limit. Orientation/Residual weights are never calibrated.

Intermediate XY-calibrated candidates may regress. Every completed cycle
records XY RMS, H/V error, angle/Keep quality and lost successful scenes/pairs
in `candidate_history` and the observer log. It does not replace the original
model/optimizer/RNG rollback anchor, even if some scores improve.

Approval requires at least one feature update and subsequent XY recalibration,
then all original diagnostic requirements relative to the original baseline:
5% linear RMS reduction, lower H/V error, direction/Keep non-regression, and
preservation of previously successful scenes/pairs. A qualifying candidate
ends immediately. Otherwise exploration continues within the fixed budget.
At the update cap, timeout, nonfinite state, solver failure or integrity error,
the changed state is rejected and the original anchor is restored. A paused
checkpoint is an unapproved diagnostic continuation, never a formal model.

Pending journals are resolved before another update. Evaluation interrupted
before phase publication replays from that journal; completed phase checkpoints
retain candidate history without duplicate decisions. Tests compare uninterrupted
and resumed weights, optimizer, RNG, counters, metrics and candidate history.

## Validation boundary

Nine new CPU lifecycle tests pass: provisional regressions, eventual approval,
final rejection/rollback, interruption at candidate/feature/phase boundaries,
nonfinite rejection, deadline rollback and separate approval requirements.

A separately authorized real-data CPU preflight must finish initial XY fitting,
exactly one shared-feature update, and XY refitting, with fixed other Heads and
Goal Encoder, before any GPU run. Its paused state is not a GPU initialization:
the one approved GPU diagnostic always starts independently from the original
600-update checkpoint. Preflight failure forbids GPU execution.

Local CPU evidence: `artifacts/recovery_diagnostics/xy_window_validation/`.
