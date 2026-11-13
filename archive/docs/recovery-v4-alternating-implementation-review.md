# Bounded v4 alternating optimization — implementation review

## Scope and current status

This opt-in path implements a new, separately authorized diagnostic starting
from the immutable 600-update Coarse checkpoint. It does not resume or extend
that terminal run. No actual feature-training trial or GPU execution was run
for this implementation. Architecture, Dataset, Teacher, losses and inference
are unchanged. Shipped config and approval example both disable training.

Source:
`artifacts/recovery_runs/recovery_v4_coarse_extend_20261108T074147Z/oracle_layout/checkpoint-000600-0019.pt`

SHA256: `5e892f898fc028a54821a0d5898367da593a3aa312fbe37f619faa3ec149dfa5`

Source commit: `5cc51dd9fafe7eec9c38a21bc9639b55704e7888`.
Original cursor (600 updates, epoch 59, batch 10, 147.900558 seconds) is retained
as provenance. The new trial has its own 10 feature-update / 120-second budget.

## Execution and freeze contract

1. Record real-forward baseline metrics and the full-set linear XY RMS.
2. Collect all 76 Train scenes, equally weighted per scene regardless of object
   count, through the existing production forward feature hook.
3. Calibrate one global XY readout using `SVDXYFitter` (`svd_huber_v1`), then one
   global orientation-logit readout using the existing CE `HeadFitter`.
4. Evaluate the fully calibrated actual forward. Check regression and the trial
   Gate; terminate immediately on PASS.
5. Freeze the **entire** final Linear (XY, logits, residual) and all non-Pose
   parameters. Update only the shared Pose feature network once, using the
   unchanged three-term pose loss accumulated over the complete 76-scene stream.
   Batches are memory chunks, not separate updates; weight each by scenes/76.
6. Invalidate the feature cache, recalculate features, recalibrate both heads,
   and evaluate again. Never accept an uncalibrated feature update as PASS.

The shared-only optimizer is fresh AdamW (existing LR `1e-4`, decay and clipping)
for this new algorithm. No coarse Head optimizer leaves can overwrite calibrated
weights. The original optimizer is preserved in the source; subsequent trial
Resume restores the new shared optimizer exactly. There is no LR sweep or
automatic extension. All Head candidates/closures and feature updates are counted
separately; the total wall budget includes cache preparation, Solver, diagnostics,
evaluation and storage. Existing 10-second solver-only per-head and 50-candidate
caps are retained. Ten seconds of the total is reserved for finalization.

## Trial Gate (not the final Arrangement Gate)

Only a completed XY + orientation calibration may pass:

- full-set retained-rank SVD RMS <= `23.453627479249874 * 0.95` mm
  **and** <= 95% of the independently reproduced starting RMS;
- H/V relative-delta error strictly lower than its starting value;
- orientation P95 does not increase, Bin correct count does not decrease;
- Keep XY and angle RMS do not increase and every previously correct Keep object
  remains <=1 mm / 1 degree;
- retain previously successful scenes/pairs, Keep-violation count and previously
  correct orientations, using the existing integrated regression checks.

The RMS computation reuses `coarse_extension.readout_diagnostic`: FP64 SVD with
FP32-relative rank cutoff, identical scene weighting and all valid objects.
The actual baseline must reproduce within 0.01 mm before any solver update.

For H/V pair (a,b), align source object IDs, compute each layout's relative
coordinates by subtracting its own centroid, then compare the predicted b-a
change to the target b-a change. Report the square root of the mean squared
Euclidean error, equally weighted across objects within each pair and then
across H/V pairs. Inputs remain predicted Goals plus source properties; labels
are consumed only by losses/evaluation.

Loss reduction alone cannot pass. A trial PASS does not populate
`completed['oracle_layout']`; the ordinary Arrangement Gate is still evaluated
separately and is not relaxed.

## Durability, timeouts and failure

`AlternatingEngine` reuses the existing split checkpoint store, immutable anchors,
head journal format, feature-cache verification and `load_integrated`. Legacy
integrated execution remains unchanged. The new state has its own schema and
contains the phase, feature/solver counts, reference/accepted metrics, pending
head/feature decisions, calibration certificates, optimizer/RNG anchor and source
provenance.

- Head candidates are journaled before observers or phase decisions. Resume
  resolves the pending solver/phase decision before any further update.
- Feature updates are fully checkpointed with a pending evaluation before any
  observer. Resume evaluates that candidate and enters recalibration; it never
  repeats the feature update.
- Full geometry evaluation runs after both heads, rather than after every inner
  Solver iteration. Inner candidate checks still run for every candidate.
- The same frozen feature cache is reused across XY -> orientation only. Feature
  changes always invalidate it; Resume verifies producer, content hash and IDs.
- Timeout/inner nonconvergence cannot pass a partly calibrated candidate. Restore
  the last completely evaluated model, optimizer and RNG; preserve attempted
  update/time charges and explicitly record `rolled_back_to_update`.
- Terminal runs cannot resume older snapshots. Corrupt/missing cache or anchor
  fails closed. Existing atomic-save and shared-weight tamper checks apply.

## Numerical verification and limitations

CPU tests cover existing SVD-Huber convergence on an overdetermined outlier
fixture (Huber optimum differs from least squares), fixed-head feature finite
differences, a re-solved Huber profile's envelope derivative, and actual v4 Pose
FiLM parameter finite differences. Lifecycle fixtures verify uninterrupted vs
crashed/resumed exact model, optimizer, RNG and update counts, including pending
final-head PASS/rollback, nonempty-optimizer rollback, cache corruption and budget.

A read-only CPU diagnostic used the actual 600-step cached features (76 scenes,
215 objects). No shared features or source artifacts were updated; diagnostic
Head coefficients were not saved. Twenty Head candidates reduced Huber loss
4.36887264 -> 1.18923724 (deployed FP32) in about 0.10 solver seconds, then hit the
existing FP32 numerical-stall guard. The projected FP64 gradient infinity norm
was `3.281e-4`, **not exact convergence**. This path retains the accepted Head,
records the certificate, and implements approximate alternating optimization;
it does not claim an exact Variable Projection gradient on this real dataset.
The finite-difference profile test establishes the envelope result on a converged
fixture, not convergence of this real-data inner solve. Whether the approximate
real-data iteration improves RMS and H/V behavior remains unverified.

CUDA performance/reproducibility and the actual 10-update trial are not tested or
authorized by this change. Do not infer generalization from this Train-only Gate.

## Future explicit execution authorization

Review `configs/recovery_v4_alternating_review.json` and the matching approval
example. A future execution must use an approved config/receipt pinned to the
implementation commit, config hash and dataset fingerprint, with explicit source
transfer approval. The source validator checks the old full checkpoint contract,
source SHA, semantic code hashes, source status and allowed config changes.
Only `oracle_layout` is allowed; Micro, extensions and the all-stage runner are
not supported by this path. Use the existing driver with `--parent-checkpoint`
for the pinned 600 source, or `--resume` only for an interrupted alternating run.

## Validation results for this change

- Related CPU regression selection: **115 passed, 48 explicit CUDA cases skipped**.
  Includes alternating, integrated lifecycle/resume/efficiency, SVD,
  Coarse-extension and readout tests.
- Final alternating-only run: **26 passed**; after adding Keep-angle RMS retention,
  the five affected gate/measurement/rollback/pending-final tests also passed.
- Static compilation and `git diff --check` passed.
- Actual source-contract and CPU real-forward baseline reproduced: 76 scenes,
  35 contrasts (21 H/V); XY RMS `23.4536835873 mm`, H/V relative-delta RMS
  `31.0909692906 mm`, orientation P95 `89.9169319153 deg`, 122/215 correct bins.
- Original checkpoint SHA unchanged. No actual shared-feature update, GPU
  training, Micro rerun or data/checkpoint migration performed.

Commands (all Python in `knolling`):

```sh
conda run --no-capture-output -n knolling python -m pytest \
  tests/neural/test_recovery_alternating.py \
  tests/neural/test_recovery_integrated.py \
  tests/neural/test_recovery_integrated_resume.py \
  tests/neural/test_recovery_integrated_efficiency.py \
  tests/neural/test_recovery_svd.py \
  tests/neural/test_recovery_coarse_extension.py \
  tests/neural/test_recovery_readout.py -q
```

Local numerical reports are in `artifacts/alternating_implementation_validation/`.
These reports are diagnostic artifacts, not promotable training checkpoints.
