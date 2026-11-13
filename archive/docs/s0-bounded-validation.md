# Bounded validation before another S0 run

`scripts/validate_s0_fixes.py` implements a fail-closed, sequential admission
process. It never calls the official training runner. Every invocation requires
a new output directory and writes a timestamped, hashed `protocol.json` before
any optimizer update. Diagnostic checkpoints use a separate format, not the
official S0 resume format. Do not interpret a diagnostic GO as S0 completion.

## Zero updates

Audit the existing complete English pilot and 32-sample selection; verify the
base checkpoint's runtime-source hash and exact loading. Recompute all 32
Hungarian optima and the actual BCE derivative, including negative weight 0.1,
300 queries and four-stage averaging. Compare the cached frozen-perception
path to full model predictions. Inspect actual Target derivatives through text
projection, Arrangement and perception, including the opposing instructions.
The original family sampler and the proposed equal-family paired weighting
must be checked; paired instructions do not guarantee usable language gradients.

## Two independent micro-tests, at most 100 updates each

Both start independently from the original `last.pt`, with fresh AdamW state,
seed 1203, existing LR/weight decay/bias exemptions/gradient clipping and strict
FP32 determinism. They are small diagnostic full-batch means, not the original
16-scene S0 training schedule. Frozen perception outputs are cached only because
those weights and running statistics are held fixed; inputs remain the model's
predicted Source and all 300 tokens. No GT is inserted into inference.

1. Detection trains only the existing final 80-class affine classifier. Five
   scenes cover low-confidence true objects and duplicate detections:
   `scene_0000_primary`, `scene_0044_primary`, `scene_0070_primary`,
   `scene_0065_primary`, `scene_0092_primary`. Use the existing Detection loss;
   Target gradients are excluded to isolate confidence optimization. Require
   5/5 correct counts, independent FN=FP=0, every matched confidence >=0.55,
   every unmatched confidence <=0.45 at the fixed threshold 0.5. All-32 count
   accuracy must not regress. Geometry and all frozen state must be unchanged.
2. Only if Detection passes, train the existing text projection and Arrangement
   on the two `scene_0051` instructions. Perception and MiniLM remain frozen.
   Use the mean of the existing Target loss on both instructions in each update.
   Require both counts correct, FN=FP=0, each instruction's Target position P95
   <=1 mm and orientation P95 <=2 degrees, and every object changing orientation
   by >=86 degrees between horizontal and vertical instructions.

Evaluate every 10 actual optimizer updates. Each micro-test must retain all
its criteria at three consecutive evaluations, including update 100. Lower loss
does not admit the next stage. Failure stops the sequence: do not try other
learning rates, losses or architectures inside the same budget.

## Joint pilot, conditional and at most 200 updates

Admission requires both micro gates. Compose their disjoint parameter changes
over the original base checkpoint, reset AdamW, and train the actual complete
RGB/text model with the unchanged Detection+Target criterion. Include both
members of each existing counterfactual family, each with half its family's
weight; singleton families retain unit weight. Group 16 families per effective
update (tail normalized separately), micro batch <=2. This differs explicitly
from selecting one family member per epoch. All 32 records remain unchanged.

Evaluate every 20 updates. Require three consecutive retained metric passes:

- 32/32 exact counts; independent FN=FP=0.
- All-93 conditional Source center/size P95 <=1 mm, angle P95 <=2 degrees.
- All-93 conditional Target center P95 <=2 mm, angle P95 <=4 degrees.
- `scene_0051` retains the tighter micro language gate.
- Task success >=75%, with no Full/Partial/No-op Task-success regression.
- Finite losses, gradients and optimizer outputs; strict determinism remains on.

These are pilot-admission criteria for considering a full S0 run. The official
S0 thresholds/evaluator remain unchanged, and both official and conditional
statistics are saved separately. Numerical/CUDA failure stops immediately.

## Execution and preservation

Run inside a fresh persistent tmux session after reviewing the protocol:

```sh
cd /workspace/knolling_bot_project
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
/venv/knolling/bin/python -u scripts/validate_s0_fixes.py \
  --checkpoint artifacts/s0/runs/EXISTING_RUN/last.pt \
  --output artifacts/s0_validations/NEW_DIRECTORY \
  --execute --register-terminal-notifications
```

The script hashes existing datasets, S0 runs and diagnosis reports before/after.
It records every actual optimizer step, evaluation, disposable checkpoint and
the final decision. CLI exit 0 means pilot GO, exit 2 means metric NO-GO;
execution errors also return nonzero. Completion/failure notifications use only the existing
`/opt/knolling-notify/notify.py watch` dispatcher and durable registration.
Registration refuses to replace a live experiment. No progress worker, direct
Telegram transport or command-listener restart is introduced.

After either NO-GO or GO, stop for human review. No automatic full S0, S1,
Teacher invocation, dataset expansion or checkpoint overwrite is implemented.
