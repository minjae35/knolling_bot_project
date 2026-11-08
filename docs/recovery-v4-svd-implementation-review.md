# v4 SVD-Huber XY calibration: implementation review

Implemented Stage 1 code and CPU/CUDA verification only. No H/V retraining,
Coarse/Fine experiment, Oracle Layout or full-76 training was started. Existing
dataset, checkpoints and caches were read-only. No training approval was created.

## Integration and unchanged behavior

`recovery_v4_svd_review.json` opts into `integrated_pose.xy_solver=svd_huber_v1`.
It remains `review_only`, `allow_training=false`, with the existing iteration/time
caps. Existing configurations default to `lbfgs_v1`; old saved optimizer states
are not silently migrated. Explicitly mixing the two solver states is rejected.
The existing code/config/parent approval checks remain in force.

Both `head_xy` and `final_xy` use `SVDXYFitter` through `make_head_fitter`.
Orientation keeps the existing CE/L-BFGS implementation. Architecture, source
inputs, predicted Goal, Dataset/Teacher, loss weights and Gate definitions do not
change. Training-only calibration copies the accepted XY rows/bias into the same
38-output Linear. Inference has no solver, planner or extra inputs.

## Numerical contract

For augmented features X=[h,1], scene/object weights D and physical-coordinate
head B0, compute D^(1/2)X=U S V^T on CPU FP64. Keep singular values above
`max(rows, columns) * eps(float32) * S[0]`, since features and deployed weights are
FP32. This fixed policy is versioned; it is not a hyperparameter sweep. Save
P=V_r/S_r and parameterize B=B0+0.010 P Z.

The objective is exactly the existing mathematical objective:
`sum_i w_i sum_j smooth_l1((X B - target)_ij / 0.010, 0)`.
Weights multiply Huber outside its clipping threshold. In its quadratic region,
the retained-coordinate Hessian is identity. No normal-equation inversion, extra
regularizer, new loss term or changed coordinate scale is used.

The first bounded candidate is the minimum-change weighted LS solution in these
coordinates. It counts as **one optimizer iteration** in the driver. Accept it
only if Huber and the mapped FP32 head do not regress; otherwise retain Z and
execute a bounded L-BFGS Huber step in the same coordinates. Subsequent steps use
that CPU FP64 Huber solver. Rank-deficient/overdetermined fixtures verify this is
not an unconditional replacement of Huber with MSE.

Factorization is feature-preparation overhead, included in the global wall-time
budget; candidate/closure work uses the existing solver timer. Preparation never
changes model weights. Deadline/stop checks surround trials, and accepted
candidates pass through the existing update counter, durable Journal and pending
evaluation/Gate/regression decision before another update.

`measurements()` evaluates the deployed 38-row FP32 Linear on cached features,
not a differently shaped two-row GEMM. The full actual Forward evaluator remains
authoritative: cache batches/padding can still cause small FP32 differences.
Nonfinite or FP32 objective-regressing candidates fail transactionally. The
cast-loss comparison permits an absolute 1e-7 roundoff margin; no position or
geometry Gate is relaxed.

## Checkpoint, resume and rollback

SVD solver state includes schema, cache key, B0, P, LS initializer, current Z,
warm-start consumption, factor hash, spectrum/rank diagnostics and complete
CPU L-BFGS state. Restore uses the stored basis rather than recomputing an SVD
whose signs or basis could differ. It verifies factor hash, dimensions/dtypes,
whitening against the pinned cache and agreement with the restored FP32 Head.

An aborted trial restores Z, warm-start flag and optimizer state before any model
publication. The existing immutable Journal/base/Anchor chain and lightweight
checkpoint format are reused. A recovered pending candidate is evaluated first;
PASS ends immediately and regression rolls back. Failed rollback artifacts
retain the existing non-resumable semantics. Fine retains its existing Adam
reset policy, invalidates features and obtains a fresh SVD basis for Final XY.

Each readout diagnostic also records the full numerical-rank and retained-rank
weighted least-squares RMS lower bounds. A bound above 1 mm proves that fixed
features cannot give every object <=1 mm error; a bound below 1 mm does not prove
the maximum-error or Knolling Gate passes. These are diagnostics, not new Gates.

## Verification

All Python ran in `knolling`; CUDA tests used the available NVIDIA RTX 4090.
Preflight disk availability was approximately 43 GiB.

The related regression suite passed 177 tests. Two additional CPU/CUDA outlier
cases were then added; the final SVD suite passed 23 tests (including overlap).
Total: **179 unique passing cases**.

- New CPU/CUDA fixtures: exact SVD fitting, FP32 forward parity, frozen non-XY
  rows, finite-difference gradient checking, unequal scene weights, rank-deficient
  outliers that reject an inferior LS/Huber candidate, iterative solver resume,
  aborted first/subsequent trials, factor/cache tampering and legacy separation.
- Synthetic integrated runs compare uninterrupted execution with crashes after
  preparation and after accepted-candidate Journals. PASS, regression rollback
  and continuation preserve accepted update counts, model, optimizer, RNG,
  cursor, solver and phase/reset state. Fine creates a distinct feature cache.
- Actual existing-cache check:
  `artifacts/svd_implementation_validation/20261108_review/result.json`.
  `scripts/validate_recovery_svd.py` creates a new diagnostic directory, applies
  one analytic CPU SVD candidate in memory, forbids actual-model
  `optimizer.step`, and evaluates Forward/Backward plus solver restoration.

| Actual Forward after analytic candidate | CPU | CUDA |
|---|---:|---:|
| Maximum position error | 0.00060345 mm | 0.00016660 mm |
| Maximum orientation error | 0.170816 degrees | 0.170816 degrees |
| H/V change error | 0.203029 degrees | 0.203029 degrees |
| Knolling Validity | 2/2 | 2/2 |
| Micro metric Gate | PASS | PASS |
| Finite Pose gradient tensors | 36 | 36 |

CUDA peak allocated memory was 632,983,552 bytes. All non-XY model tensors and
rows remained equal to the original. Checkpoint SHA256 remained
`d033eca6b347c9454059d40e259fabf6a5ce3a01dca158a6028f7e44fec4c1d5`;
cache SHA256 remained
`e9e5085ea5f66f27b763faed7bd9a5f5392da103944e96862efbbb44683adb0e`.
No new trained model checkpoint was written or promoted. The source was a
terminal `budget_exhausted` artifact used for offline verification, not an
authorized resume of its exhausted training budget.

## Review boundaries before experiments

The result establishes implementation/forward accuracy on the existing H/V
features; it is **not** evidence of a successful fresh H/V training run or of
full-dataset performance. The earlier full-76 frozen-feature analysis found an
approximately 27.71 mm RMS lower bound, so Head optimization alone cannot repair
that existing Micro-trained representation. No full-76 fitting was repeated here.

Independent review should focus on the saved basis/coordinate-to-head mapping,
transactional candidate and pending-Journal recovery, FP32 cast acceptance, and
the explicit new configuration/legacy-state boundary. New experiment approval
and a fresh receipt are still needed before either training stage starts.
