# Full-76 relative task learning

## Scope

Start from the exact validated `a037610` split candidate recorded in
`configs/recovery_v4_relative_task.json`. Keep the original Goal Encoder,
MiniLM, Perception, two Set blocks per path, object interface, angle bins and
neural inference unchanged. The two paths are now independently trainable;
their features and final heads have separate AdamW optimizers and gradient
clipping. Goal/MiniLM/Perception stay frozen. No calibration or solver runs.

## Objective

Use the existing absolute XY Huber, orientation CE and residual Huber losses.
Add two unit-weight terms with the same 10 mm Huber scale:

1. Relative XY: for each scene average Huber over unordered valid object pairs,
   comparing `p_i - p_j` to `y_i - y_j`; then average scenes.
2. Counterfactual relative XY: for each of the 35 verified command pairs, align
   object IDs and compare `(p_ai-p_aj)-(p_bi-p_bj)` to the corresponding target
   difference. Average object pairs, then command pairs.

Relative terms are invariant to a common translation. Absolute XY supervision
still determines position and supervises Keep objects. Region commands with
unchanged relative arrangements continue to receive absolute supervision.
Pair identities, target poses and labels are used only by losses/evaluation.
Forward receives the existing object attributes and predicted language Goal.

## Single approved run

`scripts/run_relative_task.py` executes only Arrangement, once. All 76 scenes
and 35 counterfactual pairs contribute to each update. Use FP32 to preserve
the validated XY readout's precision; cached frozen text avoids repeated
MiniLM execution. Fixed cosine LR goes from 1e-4 to 1e-6 over 1,200 updates,
AdamW weight decay is 1e-4, and each branch's norm is clipped at 1 independently.
There is no hyperparameter search or automatic retry.

Maximum 1,200 optimizer updates and 1,200 seconds including setup and regular
evaluations; reserve the last 30 seconds for final evaluation/reload. Evaluate
every 50 updates and at the end. Record all update losses and gradient norms,
full scene predictions, failure types and intermediate checkpoints. Temporary
regression never stops training. Nonfinite state, storage/integrity errors
stop the single run. Full 76/76 Validity and 35/35 contrast success can end early.

Best is selected lexicographically by successful scenes, successful contrasts,
fewer Keep violations, then lower XY/angle P95. Initial state is a candidate,
so a run cannot erase the starting checkpoint. The report distinguishes task
improvement from full task completion. These are training-set results, not
held-out generalization evidence.

## Persistence and verification

The existing atomic lightweight checkpoint store saves both pose paths, both
optimizer states, RNG, update/time counters, full-batch record order, pair
definitions, config and dataset/source provenance. Common immutable MiniLM/
Perception weights remain shared. Old checkpoints are read without modification.
The new versioned payload uses `relative_task.restore`; it does not pretend to
be an old frozen-orientation diagnostic checkpoint. There is no automatic
training resume/extension CLI. The saved states permit a future explicitly
authorized continuation. Best is reloaded and evaluated identically.

CPU verification passed before training: 11 unit tests including existing
split tests, independent branch gradients, relative loss finite differences,
ID joins, independent freezing, next-step optimizer/RNG checkpoint reproduction.
Actual 76-scene CPU Forward/Backward and checkpoint reload passed with zero
optimizer steps in 18.57 seconds. Evidence is under
`artifacts/recovery_diagnostics/relative_task_validation/cpu-full76/`.
