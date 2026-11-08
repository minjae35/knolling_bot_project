# Full-76 split-relative task result — 2026-11-08

One GPU run completed at the 1,200-update cap in **285.481 seconds**, including
setup/evaluation/reload. Code: `07885ec` (model/loss commit `747661a`), branch
`codex/v4-relative-task`. RTX 4090, `knolling`, FP32, peak allocated CUDA memory
1,093,172,224 bytes. No Micro, calibration, solver, retry, extension or subsequent
training ran. CPU prerequisites passed: 11 unit tests plus full-76 actual
Forward/Backward/checkpoint verification, with no real CPU optimizer step.

**Task improvement: PASS. Full 76/76 and 35/35 objective: NOT MET.**
All figures below evaluate the same 76 training records; generalization is
unmeasured. Best is update 1,200, using the predeclared Validity-first ranking.
Checkpoint reload reproduced the entire evaluation dictionary exactly.

| Measurement | Initial split model | Best |
|---|---:|---:|
| Knolling Validity | 4/76 | **18/76** |
| Counterfactual success | 0/35 | **2/35** |
| Actual forward XY Euclidean RMS, mm | 26.5152 | **24.8214** |
| XY P95, mm | 63.0039 | **53.8978** |
| XY within 1 mm | 42/215 | **34/215** |
| Orientation P95, degrees | 89.9169 | **0.03339** |
| Orientation within 2 degrees | 122/215 | **215/215** |
| Keep violation scenes | 26 | **18** |
| Keep-compliant partial/no-op scenes | 1/27 | **9/27** |
| Grounding Gate | PASS | PASS |

Best retains 2 of 4 previous successful scenes and gains 16. The two lost scenes
(`v3_9e356b0dae11a0e5b7311152`, `v3_a3d6fc3fc6211e89b6d5aa60`) both fail
`vertical_gap`; their respective object XY errors are 7.20/4.17 mm and
6.20/3.80 mm. This is a real local regression despite better aggregate success.
Keep improves in aggregate, but `v3_96db6857565a24525825ecc0` gains a new Keep
failure. Complete per-scene newly introduced errors are in `analysis.json`.

## Remaining failures and observed learning

Best successes by mode: full **9/49**, partial **4/18**, no-op **5/9**.
Failures overlap: vertical gap 35 scenes, horizontal gap 23, clearance 20,
Keep 18, collision 14, row alignment 13, requested region 2. Axis-alignment,
size-order and mixed-color-row error categories disappear. The two successful
counterfactual pairs concern axis alignment and selection.

| Loss (unit weights) | First update | Last update |
|---|---:|---:|
| Absolute XY Huber | 0.95319 | 0.89088 |
| Orientation CE | 1.01819 | 0.001637 |
| Angle residual Huber | 0.009081 | 0.00000824 |
| Relative XY Huber | 1.61121 | 1.18014 |
| Counterfactual relative Huber | 4.34042 | 2.47244 |

The actual curve shows large initial XY excursions, recovery as the fixed LR
decreases, and persistent relative/counterfactual error. XY branch pre-clip norm
is 231.95 at the first update and 563.31 at the last, versus Orientation
1.1301 and 0.00757. Independent clipping prevents the large XY gradient from
scaling down the Orientation update. This run demonstrates successful direction
learning and better task success; it does **not** establish that the remaining
XY problem is an architectural limit or that more updates would solve it.
Precise spacing, Keep and local XY fidelity remain unresolved. No follow-up
experiment or automatic modification was performed.

## Preserved evidence

Run directory: `artifacts/recovery_runs/recovery_v4_relative_task_once_20261108T092000Z/`

- `checkpoint-001200.pt`: Best, complete trainable branches and optimizer/RNG state.
  SHA256: `fbbb70a8354171bc453e6642d12c8c53963e064abdc33d25c67be3ba9193ea79`.
- `checkpoint-000000.pt` and every 50-update checkpoint: starting/intermediate states.
- `result.json`, `best-evaluation.json`: final summary and all actual predictions.
- `history.jsonl`, `evaluation-curve.csv`, `learning-curve.png`: actual update losses,
  gradient norms, intermediate evaluations and plotted curves.
- `analysis.json`: baseline/Best accuracy, lost/gained scenes and new errors.
- `../recovery_v4_relative_task_once_20261108T092000Z.log`: tmux/tqdm log.

Source SHA256 remains `03a12d6f135c5169ec7bb9f8560f8c216785000f519a685ed69587e01ec093d6`.
MiniLM, Goal Encoder and Perception tensors remain exactly unchanged. Existing
datasets and checkpoints were not overwritten or deleted. Plot generation
after the run used Matplotlib installed only in `knolling`; no model training
or evaluation was rerun to produce the charts.
