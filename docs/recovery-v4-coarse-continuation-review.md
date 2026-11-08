# Bounded Coarse continuation and SVD numerical stalls

This change preserves the v4 architecture, dataset, loss, LR, clipping, lightweight
storage, and existing Micro evidence. No GPU training is authorized by the shipped
configuration or receipt template.

## Numerical stall

The XY solver still rejects every candidate whose deployed FP32 Huber loss exceeds
the accepted loss by the existing `1e-7` tolerance. It classifies the rejection as a
recoverable numerical stall only when the objective and coefficients are finite,
the FP64 solver objective does not increase, and evaluating the FP32 coefficients
in FP64 also does not increase loss. NaN, infinity, or real coefficient/objective
regression remains fatal. The solver restores its prior coordinates and L-BFGS
state before returning the typed stall.

The driver publishes the unchanged accepted head and a pending phase-completion
decision in its atomic journal. It does not increment the accepted-update counter.
Evaluation, Gate and regression checks run before advancing to Orientation
Calibration. A crash after that journal cannot retry the rejected candidate on
Resume. Actual performance regression still rolls back to the phase anchor.

## Explicit Coarse extension

Use `configs/recovery_v4_coarse_extension_review.json` as the disabled template and
`configs/recovery_v4_coarse_extension_approval.example.json` as the unapproved
receipt template. After independent review and user GPU approval, create a new
runtime configuration/receipt pinned to the target commit, runtime config digest,
dataset fingerprint and source contract hashes. Launch only the existing driver
with `--stage oracle_layout --extend-from <source>`, in a new output directory.
Do not use ordinary `--resume` on the old terminal run or edit its checkpoint.

The pinned source is:

`artifacts/recovery_runs/recovery_v4_svd_arrangement_20261108T070714Z/oracle_layout/checkpoint-000200-0010.pt`

SHA256: `4dec4d191e079f893d5213262f338d3103fd5c18822a660ce1141e5eb634de68`

It contains 200 accepted Coarse updates, epoch 19 / batch 10, and 50.07741617504507
seconds of driver usage. The continuation preserves those counters, all model
weights, Adam leaves/moments/steps, and RNG. Only the in-memory phase transitions
from the untouched Coarse boundary back to Coarse. Source config/model/loss/teacher/
evaluator contracts, anchor, optimizer/head synchronization, and all three existing
Micro evidence files are verified. Original files are never changed. The ordinary
same-commit/config Resume contract and terminal-run prohibition remain unchanged.

Cumulative caps: Coarse 1000; XY 50; Orientation 50; Fine 100; final XY 50; final
Orientation 50; total 1300 accepted iterations and 600 driver seconds. Existing
usage is not reset. Fine entry and all task Gates remain unchanged.

A complete Train-only readout at source update 200 establishes the baseline. At
400/600/800/1000, diagnostics record the FP32-retained FP64 SVD XY RMS lower bound,
full pose loss components, and orientation-bin accuracy. Diagnostics do not fit
or change heads; RNG is restored. Evaluation/diagnostic/I/O overhead counts against
the cumulative driver budget. Two consecutive relative improvements below 5%
(including worsening) stop with `coarse_stalled`. A 5% or greater improvement resets
the streak. The history and stop decision are checkpointed before reporting, so a
crash cannot erase a weak interval or execute another update after a pending stop.

## Verification

CPU tensor fixtures cover rejected-candidate state restoration, finite versus
nonfinite failures, Journal crash/replay, Gate/regression handling, phase transition,
source approval/contract rejection, exact Adam/RNG/data-cursor preservation,
uninterrupted versus resumed execution, the two-interval stop, and cumulative time.
CUDA variants are deliberately disabled for this implementation-only task.

The actual 200-update checkpoint was also loaded read-only: its Micro evidence and
anchor passed validation, its SHA256 stayed unchanged, and its model, optimizer,
RNG and data/time cursor matched the returned continuation state exactly. No
production optimizer step or training run was executed.

Completed validation in `knolling` with `CUDA_VISIBLE_DEVICES=''`: **141 passed,
62 CUDA tests skipped** across SVD, Coarse extension, integrated lifecycle/Resume/
efficiency, driver/v4 driver, storage, readout, provenance, legacy extensions and
v4 runtime tests. `git diff --check` also passed. No real Micro or Arrangement run
was started.
