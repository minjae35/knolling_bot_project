# v3 Stage 1 — bounded GPU result

**Stage 1 did not pass.** The run completed 500 updates in **123.23 seconds**
against the authorized 500-update / 600-second ceiling. Teacher Fidelity and
Knolling Validity both failed. No extension, retry, Stage 2 or Stage 3 was run.

## Execution and reproducibility

- Run: `recovery_v3_stage1_20261108T032813Z`.
- Executed commit: `03134ec24bcb2ffd23f9deb5679912092ab07837`.
- Reviewed model implementation: `a069b8f`; architecture, loss, teacher and gates
  were unchanged. The execution change adds explicit `--stop-after` scope.
- Config: `configs/recovery_v3_gpu_stage1.json`. Approval receipt authorizes only
  `oracle_layout` and binds code, config and data fingerprint.
- Conda `knolling`; RTX 4090; GT objects + natural-language tokens. This run
  provides no evidence about learned perception or RGB-to-target accuracy.
- Batch 8, BF16 attention with FP32 pose heads/losses, cached Frozen MiniLM and
  source images, pinned memory and zero DataLoader workers for this tiny suite.
- Dataset: 76 records / 25 original families / 35 contrast pairs, immutable
  `artifacts/recovery_v3_verification_r1/annotations.json`.
- tmux: `knolling-v3-stage1`; reconnect with
  `tmux attach -t knolling-v3-stage1`. The session remains available after exit.
- Existing terminal dispatcher registration only; no periodic Telegram worker
  or direct transport was introduced.
- Elapsed time is the driver's clock, including its evaluations and checkpoint
  work, but excluding initial model/data setup. Exit code 2 is the expected
  unmet-gate/budget-stop result; `error` is null, not a runtime/nonfinite failure.

Preflight: real cached R18/MiniLM BF16 Arrangement forward/backward, batch 8,
finite gradients, zero optimizer updates, peak allocated memory 625.39 MiB.
Runtime/scope regression tests: **11 passed, 1 explicitly gated CUDA test skipped**.
The successful and unsuccessful Stage 1 fixtures both verify no Stage 2 launch.

## Final measured results

| Metric | Final | Criterion / interpretation |
| --- | ---: | --- |
| Teacher position error P95 | 250.50 mm | Fidelity <= 1 mm |
| Teacher angle error P95 | 89.86 degrees | Fidelity <= 2 degrees |
| Teacher Fidelity | FAIL | Complete set, coordinate criteria |
| Knolling Validity | 1/76 (1.32%) | All 76 must succeed |
| Pure-axis contrasts | 0/21 | All 21 must succeed |
| All six-factor contrasts | 0/35 | All verified pairs must succeed |
| Raw geometry success | 44/76 (57.89%) | Geometry alone does not imply task success |
| Pose-derived Move accuracy | 74.42% | Diagnostic; no Move classifier |
| Keep violations | 27 scenes | 18 partial + 9 no-op scenes |

Per-mode success: full **1/49**, partial **0/18**, no-op **0/9**. Failure counts
overlap: vertical gap 48, axis alignment 32, clearance 32, requested region 28,
Keep 27, horizontal gap 19, row alignment 18, collision 15, size order 7 and
mixed-color row 1. The independent validity gate was not relaxed.

| Updates | Mean total loss | Position Huber | Angle CE | Residual Huber |
| --- | ---: | ---: | ---: | ---: |
| 1–100 | 9.254 | 7.575 | 1.590 | 0.090 |
| 101–200 | 7.127 | 6.112 | 0.999 | 0.016 |
| 201–300 | 6.378 | 5.620 | 0.748 | 0.011 |
| 301–400 | 5.758 | 5.155 | 0.594 | 0.009 |
| 401–500 | 5.345 | 4.823 | 0.512 | 0.010 |

Loss decreased, but position P95 changed only **259.39 -> 250.50 mm** between
updates 250 and 500; pure-axis success remained zero. The last checkpoint,
not a selected intermediate best checkpoint, determines this report.

## AR comparison on the same evaluation protocol

Both preserved AR checkpoints were reevaluated without updates on the **same
76 v3 records, GT sources, strict semantic tolerances and 35 contrast pairs**.
All three use BF16 attention/FP32 pose arithmetic. The read-only AR adapter only
exposes the v3 evaluator interface; it preserves the actual AR decoder and its
hard Move/Keep output. No pose correction or target substitution is applied.

| Model | Updates | Position P95 | Angle P95 | Valid tasks | Raw geometry | Pure-axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 500 | 143.89 mm | 89.91 deg | 8/76 (10.53%) | 40/76 | 0/21 |
| AR | 1,000 | 122.15 mm | 89.98 deg | 12/76 (15.79%) | 42/76 | 0/21 |
| v3 joint | 500 | 250.50 mm | 89.86 deg | 1/76 (1.32%) | 44/76 | 0/21 |

All three fail both Fidelity and Validity; v3 does not demonstrate an improvement
in task success or position accuracy at this budget. V3 has fewer collision
scenes (15 versus 26 for either AR checkpoint), but worse Keep preservation
(27 violating scenes versus 8/9).

**This is not an architecture-only ablation:** AR was trained against the legacy
teacher, whereas v3 was trained against the new canonical teacher, and the
1,000-update AR run has twice the updates. Historical native AR results remain
in `recovery-initial-gpu-run.md` and `recovery-stage1-extension.md`; do not mix
their 96-record denominators or old targets with this common-protocol table.

## Evidence-based failure diagnosis

1. **Coarse displacement is underlearned.** Required source-to-target movement
   P95 is 300.81 mm, whereas the final checkpoint's initial displacement head
   predicts only 32.84 mm at P95. Initial target position error P95 is 302.19 mm.
   Each refinement can correct at most 50 mm per coordinate. Holding the current
   initial prediction fixed, even ideal refinements have a position-error lower
   bound of **193.31 mm P95**, computed objectwise as
   `norm(max(abs(target - initial_pose) - 0.100, 0))`.
   This identifies a concrete limitation of the *learned initial estimate plus
   bounded correction* at this checkpoint, not a proof that the unbounded
   initial head can never learn large displacements.
2. **Refinement operates but is insufficient.** Position P95 goes
   **302.19 -> 280.36 -> 250.50 mm** across initial/refinement 1/refinement 2.
   Median angle error goes **3.88 -> 1.05 -> 0.61 degrees**, yet angle P95 stays
   near 90 degrees. Only **20/860 correction coordinates** are within 1% of the
   50 mm cap; widespread tanh saturation is therefore not established as the
   sole cause. The remaining problem also includes inaccurate correction
   direction/magnitude, not just a hard range ceiling.
3. **Direction choice remains unreliable.** The final prediction chooses the
   wrong teacher orientation bin for **64/173 selected objects**. Small residual
   loss cannot repair a horizontal/vertical bin mistake. Zero pure-axis pair
   success shows that reliable command-conditioned arrangement was not learned
   within the budget; it does not prove that all language information is ignored.
4. **Keep is not yet precise enough without copying.** Objects with exact
   source=target labels have position-error P95 **15.83 mm**, against the 1 mm
   Keep criterion. Removing the hard copy branch exposed this learned-pose
   requirement; all partial/no-op scenes currently fail Keep. This does not
   justify restoring a rule-based planner or replacing predictions with labels.

The remaining position failures warrant reviewing the coarse displacement
parameterization/scale together with the bounded refinement range before simply
repeating training. Axis-conditioned bin selection and Keep precision also need
attention. These observations do **not** isolate the optimization cause of the
underlearned initial head or establish a successful remedy. No architecture,
loss-weight or budget change was made during this run, and no extra training was
used for diagnosis. Loss curves alone do not justify an automatic extension.

## Artifacts and preservation

- Launch receipt/config, console and reproducible zero-update comparison scripts:
  `artifacts/recovery_launches/recovery_v3_stage1_20261108T032813Z/`.
- Run results, `training-summary.json`, comparison metrics/predictions,
  refinement diagnosis and displacement bounds:
  `artifacts/recovery_runs/recovery_v3_stage1_20261108T032813Z/`.
- Final checkpoint: `oracle_layout/checkpoint-000500-0022.pt` in that run.
- **2,189 pre-existing files** retain their recorded size/mtime. The preserved
  AR 1,000-update checkpoint SHA-256 is
  `a5346ad7c9f8cd6c9aed14c5e64c6c76319802612d87fb5b9ef98b6ad0024984`.
- Active v3 annotations retain SHA-256
  `b42ee5b9e99afeeb37036a63f4e180b791719212faa68be226f7745beb051e7c`.
- There are no `perception`, `predicted_layout` or `joint` output directories.
  The GPU is idle; the tmux session is retained for inspection.
