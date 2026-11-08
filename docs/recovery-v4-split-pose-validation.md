# Saved-candidate split pose CPU validation

This is a diagnostic composition, not an Arrangement training PASS. No real
model optimizer step, solver fitting, or GPU execution is needed or performed.

## Minimal change

`recovery.split_pose.SplitJointPose` keeps the two-update candidate's XY feature
path and head under their original parameter names. A frozen copy of the
original 600-update pose path supplies orientation logits and residuals. Both
receive the same object inputs and the same predicted Goal/selection; the Goal
Encoder runs once. No target labels enter either forward path.

The explicit `split_pose` checkpoint marker records both donor hashes and the
frozen branch tensor digest. Strict restoration attaches that branch before
loading. The existing lightweight store saves its parameters and buffers, with
unchanged atomic writes and common-weight integrity checks. Legacy checkpoint
loading remains supported.

Composition preserves the candidate's XY optimizer names/state, cursor and
budget. It invalidates feature/solver caches and marks the new combination for
evaluation **before any next update**. The rollback anchor contains both
original paths. The original formal Gate is unchanged. Diagnostic orientation
bin accuracy now comes from actual forward logits, rather than reconstructing
them from the XY feature cache.

## Pinned donors

- Original: `artifacts/recovery_runs/recovery_v4_coarse_extend_20261108T074147Z/oracle_layout/checkpoint-000600-0019.pt`
  SHA256 `5e892f898fc028a54821a0d5898367da593a3aa312fbe37f619faa3ec149dfa5`.
- XY: `artifacts/recovery_runs/recovery_v4_xy_window_once_20261108T084741Z/oracle_layout/checkpoint-000002-0081.pt`
  SHA256 `8f28e2f222c3c6e335ddea8b7d67c7a3a34cb63b8f468476b64a6f047be022a1`.
- Dataset fingerprint: `3af9210f8cdb5d9acda5084200940c4d2407c6a28a7268cc36f55c4793b7e6bf`.

## Same 76 records, CPU results

| Measurement | Original 600 | Split candidate 2 |
|---|---:|---:|
| Optimal linear XY RMS, mm | 23.453684 | 21.245798 |
| H/V relative error, mm (21 pairs) | 31.090969 | 26.855417 |
| Orientation P95, degrees | 89.916932 | 89.916932 |
| Correct orientation bins | 122/215 | 122/215 |
| Keep XY RMS, mm | 70.644159 | 34.006129 |
| Keep angle RMS, degrees | 24.311267 | 24.311267 |
| Valid scenes | 1/76 | 4/76 |

XY linear RMS improved 9.41%; this is the existing SVD diagnostic lower bound,
not a claim that the deployed head attains that error. Actual forward XY is
bitwise equal to the saved candidate; orientation/logits/residuals are bitwise
equal to the original. The previous successful scene and existing orientation/
Keep non-regression criteria are preserved. Individual already-failing Keep
objects are not guaranteed to improve. Counterfactual success remains 0/35.

The composed checkpoint reloads with identical actual outputs. A fresh CPU
engine resumes, evaluates the pending candidate, and returns diagnostic PASS
at inherited update 2 without any optimizer step. Model, optimizer and active
CPU RNG states match exactly. Original donor hashes remain unchanged.

## Tests and scope

- 35 existing alternating/window CPU tests and 5 split-path CPU tests passed.
- Tests cover forward branch identity, shared Goal, padding, valid-object
  permutation, target-label independence, frozen parameters, synthetic CPU
  optimizer/RNG resume equality, and branch tamper rejection.
- Real validation artifacts: `artifacts/recovery_diagnostics/split_pose_validation/`.
- GPU-donor CUDA RNG bytes remain in provenance; the derived CPU checkpoint has
  CPU-only active RNG state. This is an explicit new diagnostic artifact, not
  an in-place alteration of the GPU checkpoint. CUDA Resume has not been tested.
- Derived configuration disables training and has no training approval receipt.
  Neither the diagnostic PASS nor its saved state authorizes further training.
