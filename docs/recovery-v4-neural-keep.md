# Neural Keep intervention

## Evidence and source

Start from `codex/v4-relative-task` at `1965663`, not the failed Pairwise model.
The accepted checkpoint is
`artifacts/recovery_runs/recovery_v4_vertical_gap_once_20261108T093400Z/checkpoint-001200.pt`
(SHA256 `983a6a5fd8579edc08a10be6054c3b94b870af66db3b9215524988e0e72ed54d`).

One CPU-only intervention on its saved 76 predictions copied the full source pose
for the 60 GT Keep objects. No model training or additional forward was used.
Validity improved **28 → 45/76**, Counterfactual **3 → 10/35**, with no lost
successful scenes. Remaining overlapping failure counts: vertical gap 18,
horizontal gap 17, clearance 10, collision 7, row alignment 4, size order 1,
region 1. This establishes a useful Keep bottleneck but leaves Move layout errors.
The exact intervention and outputs are preserved in
`scripts/diagnose_oracle_keep_once.py` and
`artifacts/recovery_diagnostics/oracle_keep_once/result.json`.

18 of the 60 Keep objects are selected objects already arranged correctly.
Consequently, complementing the Goal Encoder's selection would be incorrect.

## Minimal implementation

`NeuralKeepPose` reuses the accepted Set features and orientation branch. Its
single shared MLP takes each object's normalized XY Set feature, the masked
scene mean, physical source attributes, predicted Goal vector, and predicted
selection logit. A scalar logit >= 0 selects the existing neural target pose;
a negative logit copies the source XY and orientation. No target label or
planner enters forward, including the unchanged Perception object interface.

Only the new classifier learns. Goal Encoder, MiniLM, Perception, XY and
Orientation weights are frozen. Scene-averaged masked binary cross entropy
supervises `not copy_required`; padding contributes no loss. Hard routing
does not require gradients through the decision because its classifier has
direct supervision. The initial zero final layer routes all objects to Move,
preserving the accepted model before training. Orientation logits/residuals
describe the Move proposal; `xy` and `angle` are the final routed pose.

The existing full-76 driver, seed 1203, AdamW and cosine learning-rate schedule,
gradient clipping, lightweight atomic checkpoints, RNG and optimizer storage
are reused. The Keep mode overrides the old pose-loss objective with BCE only;
the legacy config fields identify the frozen donor contract. No SVD or
calibration runs. The driver remains a single-run launcher, not an automatic
resume or retry service; checkpoints support exact state restoration.

## Validation and single GPU run contract

20 CPU tests pass, covering routing, gradient isolation, permutation, padding,
supervision isolation, exact checkpoint/optimizer/RNG next-step restoration,
and the existing relative/split/gap functionality. A real full-76 CPU check
passed with zero optimizer steps, baseline 28/76 and 3/35, and exact output
reload (`artifacts/recovery_diagnostics/neural_keep_validation/cpu-full76`).

`configs/recovery_v4_neural_keep.json` pins checkpoint and Oracle hashes.
One RTX 4090 run is capped at 1,200 updates / 1,200 seconds, with evaluation
every 50 updates and a 30-second finalization reserve. Best selection requires
both task counts to improve, all old successful scenes to survive, no increase
in Keep violations, and no orientation regression. Temporary regressions do
not stop training. Nonfinite state or integrity failures do stop it.

Early completion is allowed when every object is correctly classified and the
actual neural forward reproduces at least Oracle's 45 scenes / 10 pairs with
the same nonregression checks. Reload Best and independently evaluate before
reporting success. Accuracy on these 76 training scenes is not generalization
or proof of completed RGB-to-pose training.
