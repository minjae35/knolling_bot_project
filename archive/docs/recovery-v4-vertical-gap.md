# One task-directed vertical-gap refinement

Baseline is the immutable Best from `35ce11f`: 18/76 Validity, 2/35 contrast
success, orientation P95 0.03339 degrees. No new model inference was required
to diagnose failures: all 76 saved predictions were independently rechecked
against the existing geometry/semantic evaluators.

## Evidence and single hypothesis

58 scenes fail; 29 have overlapping errors. Counts (not additive) are vertical
gap 35, horizontal gap 23, clearance 20, collision 14, Keep 18, row alignment
13 and requested region 2. Exclusive failures are vertical gap 13, horizontal
gap 5 and Keep 11. The union of horizontal/vertical gap failures is 45 scenes.
Predicted row membership differs from Teacher membership in 10 scenes.

For the 52 Teacher-adjacent row boundaries, absolute vertical-gap error has
median 4.34 mm and P95 44.52 mm: 26 gaps are too small and 10 too large by
more than the evaluator's 2 mm tolerance. This is not just global translation.
An offline diagnostic substitution of correct common centers yields 25 valid
scenes; substituting only correct relative Y while retaining predicted X and
common Y offset yields 36. These use labels only for causal localization and
are **not** model results, a planner or inference code.

Selected hypothesis: directly supervising the actual adjacent-row boundary
gap can improve the largest isolated task failure. This is a testable
hypothesis, not proof that the network will learn it within the approved budget.

## Minimal change

Keep both Set feature paths, Goal Encoder, dataset and Perception interfaces.
Freeze the complete trained Orientation path. Add one unit-weight Huber term
to the existing five losses: adjacent vertical **edge gap** error divided by
0.002 m, the existing task evaluator tolerance. A rectangle's vertical half
extent is `(a*abs(sin(theta)) + b*abs(cos(theta)))/2`. The predicted gap is the
upper row's minimum bottom edge minus the lower row's maximum top edge.
Teacher row memberships and target gaps are supervision only. This continues
to supervise a boundary when a bad prediction merges rows. No-op/single-row
scenes have no added gap term. Existing absolute XY supervision remains for
Keep and common translation; all existing relative/contrast terms remain.

The gap term detaches angle, and the frozen path has no optimizer. Hashes cover
Goal, MiniLM, Perception and the entire Orientation path. Reload restores its
frozen flags. All old configurations retain their previous behavior.

Use the same seed, FP32, full-76 batch, optimizer, clipping, weight decay and
fixed cosine LR schedule; no calibration or hyperparameter search. Start from
the pinned previous Best, with fresh XY Adam state for the changed objective.
Run once, at most 1,200 updates / 1,200 seconds. Intermediate regressions are
recorded, not used as a stop condition. Nonfinite/integrity failures stop.

Best selection first prefers models satisfying **both** Validity >18 and
Counterfactual >2 with identical per-object orientation error evidence. Within
that eligibility class use the existing Validity-first ranking. The initial
model and every 50-update checkpoint remain preserved even if not adopted.
Final evaluation compares actual predictions and each failure category.

Evidence: `artifacts/recovery_diagnostics/task_gap_analysis/baseline.json`;
CPU tests: `tests/neural/test_recovery_vertical_gap.py` plus prior relative/split
tests. All 15 passed, including edge geometry, gradient finite differences,
frozen orientation after XY update and checkpoint restoration.
Actual full-76 CPU Forward/Backward/checkpoint verification also passed in
18.88 seconds with zero optimizer steps. The baseline remained 18/76 and 2/35;
the new gap loss was 4.50072 and Orientation received no gradient.
