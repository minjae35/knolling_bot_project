# v4 Keep-all and axis Micro results — 2026-11-08 UTC

Keep-all passed. Horizontal/Vertical failed at its update cap; execution stopped.
No automatic extension, repeated trial, or Oracle Layout training was performed.

## Execution contract

- Code: `5078a40850174b85750ab0731db07ed5a05d2709`, identical training code to
  the passing translation run at `fa60fdb` (only documentation changed).
- Conda `knolling`, RTX 4090, FP32, seed 1203; existing v4 architecture/loss/data.
- Each case independently loaded the same passed 450-update Grounding checkpoint,
  SHA256 `b1191dd1daaec602b366fba68cbd222c4031337130e4c981e0c2481bc01d0406`.
  Each started with a fresh Pose optimizer; neither inherited another Micro model.
- Goal Encoder, MiniLM, and Perception remained frozen. Only Pose was trained.
- Per-case cap: 200 updates / 120 seconds; evaluate every update and stop on Gate.
- Precision transition: max position <=10mm AND max angle <=2 degrees; all Pose
  LR 1e-4 -> 1e-6; reset first moment once, retain second moment and optimizer step.
- Existing Gate: all valid objects <=1mm position and <=2 degrees angle, plus
  Knolling Validity. Axis additionally requires <=4 degrees pair-change error.

## Results

| Metric | Keep-all | Horizontal/Vertical |
|---|---:|---:|
| Samples / objects per sample | 1 / 3 | 2 / 2 |
| Status | PASS | FAIL: update budget exhausted |
| Updates | 91 | 200 |
| Driver elapsed seconds | 15.917 | 41.005 |
| Precision transition update | 75 | Never qualified |
| Final max position error, mm | 0.903909 | 14.266067 |
| Final max angle error, degrees | 0.083069 | 89.956781 |
| Axis pair-change error, degrees | N/A | 89.919871 |
| Knolling Validity | PASS | FAIL for both commands |
| Keep violations | 0 | 0 |
| First-25 mean loss | 8.00712 | 10.38345 |
| Last-25 mean loss | 1.38503 | 1.46528 |

Keep targets were verified equal to source position/orientation. It passed at the
first passing evaluation; the next independent axis experiment then started.
Axis used both instructions on the same source in each update. Its best measured
max position error was 11.936788mm at update 198. Its max angle error stayed near
90 degrees throughout (best 89.163352 degrees at update 3). Consequently the
precision transition did not activate; this is correct policy behavior.

## Axis failure: observed evidence versus hypothesis

Read-only CPU forward/backward diagnostics used the final checkpoint, with no
optimizer constructed or stepped. Source features are identical for the pair.
Teacher angles are [0,0] for horizontal and [-90,-90] for vertical; the goal
vectors differ (L2 distance 1.41076). Both sample-level Goal evaluations pass
(8/8 correct each; minimum true probability 0.993934 / 0.994800).

The Goal Encoder's axis probabilities are 99.8307% horizontal and 99.9451%
vertical. The fault is therefore downstream of correct axis grounding in this
case. Final Pose angles, in degrees, are approximately:

| Command | Object 1 | Object 2 |
|---|---:|---:|
| Horizontal | -0.029 | -89.957 |
| Vertical | +0.058 | +89.963 |

Angles are pi-periodic: -90 and +90 denote the same orientation. Each object
therefore retains almost the same orientation under either command. The actual
changes are only +0.08694 and -0.08013 degrees instead of 90 degrees. Both samples
select bins [9,0], while correct bins are [9,9] and [0,0]. True-bin probabilities
are only 0.43–0.48. Bin selection has not learned the conditional distinction.

Swapping the predicted Goal vectors changes positions by up to 20.184mm, but
angles by only 0.08694 degrees. Goal conditioning is connected and partly used;
it is not globally ignored. FiLM weights receive finite, nonzero angle gradients
(block 0 norm 0.13525; block 1 norm 0.12270). No detached conditional path or
incorrect freeze was found in these checks.

At the final checkpoint:

| Loss | Value | Pose gradient norm |
|---|---:|---:|
| Position Huber | 0.651309 | 326.621 |
| Angle CE | 0.775022 | 1.06177 |
| Angle residual Huber | 0.0000298 | 0.05121 |

Position/angle-CE gradient cosine is -0.06999. All 200 updates clipped gradients.
This is evidence of a large gradient-scale imbalance and incomplete conditional
orientation learning, not proof that clipping or a particular module alone
caused failure. Adam normalization prevents equating raw gradient ratios with
effective update ratios. Loss decreased, but the task-critical axis distinction
did not improve; merely extending this trajectory is not justified by that loss
decrease. Capacity limitations versus optimization under the existing objective
remain unseparated by this single bounded experiment.

Position failure also includes relative-layout errors (max centered error
8.38/8.62mm), not just a common shift. Geometry checks report axis alignment,
collision, clearance, and vertical-gap failures for both instructions. This is
broader than the previously solved translation precision issue.

The next review should focus on learning the Goal-conditioned orientation bins
and balancing position/orientation optimization, before requesting any new
training budget. No architecture, loss, threshold, or optimizer changes were made
in response to the failure during this run.

## Preservation and artifacts

Both anchors match the Grounding parent's weights bitwise. Only Pose state
tensors changed; frozen modules match their anchors. Optimizer steps are 91 and
200, respectively. Saved phases are fine (transition 75) and coarse (no reset).
All 2,340 pre-existing artifact files retained size and mtime; the Grounding
parent hash was also verified. CPU auditing did not update any model.

Run root: `artifacts/recovery_runs/recovery_v4_micro_remaining_20261108T044929Z/`.

- `keep/checkpoint-000091-0005.pt`, SHA256
  `cabe02c5cf6d7344842c6821f3dadff7d7ada0ddd3d66deffea78975c4ddc98e`.
- `axis/checkpoint-000200-0010.pt`, SHA256
  `11adeba48c75538cabcbe1e1318ea7307e2a8086f8ff79fc277c6d1c7546a19f`.
- Each case has `history.jsonl` with every update's loss, evaluation, and poses.
- `analysis.json` includes per-update centroid errors/movement and preservation
  audit; `axis-diagnostics.json` contains actual predictions, Goal probabilities,
  loss/gradient values, and the read-only Goal-swap diagnostic.
- Approval receipt/config/launch scripts are preserved under the same run name
  in `artifacts/recovery_launches/`.

The completed tmux session is `knolling-v4-micro-remaining`:
`tmux attach -t knolling-v4-micro-remaining`.

Overall Micro status: translation PASS, Keep-all PASS, axis FAIL. The prerequisites
for full Oracle Layout are not satisfied, and it remains unstarted.
