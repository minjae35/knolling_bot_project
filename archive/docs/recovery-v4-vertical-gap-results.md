# Vertical-gap refinement: adoption PASS, full task incomplete

One full-76 GPU run on code `a2e50f9` (loss/freeze implementation `5bd4507`)
completed **1,200 updates in 324.960 seconds**. Best is update 1,200. The
predeclared adoption criterion is met: Validity >18, Counterfactual >2 and
unchanged Orientation. Actual checkpoint reload reproduced all metrics
exactly. No retry, Micro, calibration, additional training or parameter search
was performed. CPU prerequisites passed: 15 unit tests and actual full-76
Forward/Backward/checkpoint validation with zero real CPU optimizer steps.

## Outcome on the same 76 training records

| Metric | Previous Best | New Best |
|---|---:|---:|
| Validity | 18/76 | **28/76** |
| Counterfactual | 2/35 | **3/35** |
| Actual XY Euclidean RMS, mm | 24.8214 | 27.7162 |
| XY P95, mm | 53.8978 | 62.6479 |
| XY within 1 mm | 34/215 | 27/215 |
| Orientation P95, degrees | 0.03339065 | **exactly unchanged** |
| Orientation within 2 degrees | 215/215 | 215/215 |
| Keep violation scenes | 18 | 20 |

All saved per-object predicted orientation values are exactly unchanged, as
are the complete Orientation, Goal, MiniLM and Perception weights. These are
training-set measurements; held-out generalization was not evaluated.

## Each failure category (overlapping counts)

| Failure | Before | After |
|---|---:|---:|
| Vertical gap | 35 | **21** |
| Horizontal gap | 23 | **19** |
| Clearance | 20 | **13** |
| Collision | 14 | **9** |
| Row alignment | 13 | **4** |
| Requested region | 2 | **1** |
| Keep | 18 | **20** |
| Size order | 0 | **2** |

The original 58 failing scenes included 29 with multiple failures. Of the 13
scenes failing **only vertical gap**, **11 now succeed**. The run gains 14
successful scenes and loses 4, for a net gain of 10. Success by mode changes
from full 9/49 to 21/49, partial 4/18 to 4/18, and no-op 5/9 to 3/9.

Lost previous successes:

- `v3_5cb4ed6f4dd06c17b2a7459f`: no-op, Keep.
- `v3_47c62e2a4f75fc1cec683b9d`: no-op, Keep.
- `v3_09df4d3cd2c8102b4ffcaa22`: partial, Keep.
- `v3_6426f9f2c7ac4638966481ac`: full, horizontal gap.

New size-order errors occur in `v3_ab9763412e38506e17eb3528` and
`v3_d901837b50ad8c364a06a7ea`; these also retain other layout errors.

## Interpretation and limits

The targeted vertical edge-gap loss falls **4.50075 → 0.45067**, and actual
vertical-gap failures fall 35→21. The single-bottleneck hypothesis is supported
by task results, especially the 11/13 isolated cases resolved. It was not
refuted in this run. This does not prove the loss alone caused all improvement:
there is no extra continuation-only control experiment.

The broader claim that this fixes XY generally is unsupported. Absolute XY
Huber rises 0.88974→1.10969; existing relative Huber rises 1.18003→1.47094;
contrast-relative Huber changes only 2.47239→2.44132. Better local spacing
comes with worse canonical XY precision and Keep. The learning curve has large
early regressions and later task recovery as the unchanged LR schedule decays.
These observations do not establish an architecture limit. Full 76/76 and
35/35 remain unmet. No follow-up experiment or automatic fix was launched.

## Preserved artifacts

Run directory:
`artifacts/recovery_runs/recovery_v4_vertical_gap_once_20261108T093400Z/`

- Best: `checkpoint-001200.pt`, SHA256
  `983a6a5fd8579edc08a10be6054c3b94b870af66db3b9215524988e0e72ed54d`.
- Starting and every 50-update checkpoint retained with optimizer/RNG states.
- `result.json`, `best-evaluation.json`, `analysis.json`: complete metrics,
  actual predictions, retained/lost/gained scenes and failure comparisons.
- `history.jsonl`, `evaluation-curve.csv`, `learning-curve.png`: actual training
  and evaluation curves.
- Previous Best remains untouched, SHA256
  `fbbb70a8354171bc453e6642d12c8c53963e064abdc33d25c67be3ba9193ea79`.

Session: `knolling-v4-vertical-gap` (completed). Terminal log:
`artifacts/recovery_runs/recovery_v4_vertical_gap_once_20261108T093400Z.log`.
Frozen/common weights use the existing lightweight atomic storage. All old
datasets and experimental results remain preserved.
