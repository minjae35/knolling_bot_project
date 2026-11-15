# Evaluation

A scene passes only when every check passes. The official v2 numbers use `semantic_errors_v3` and `geometry` with these tolerances:

| Check | Tolerance |
| --- | --- |
| Keep position | 1 mm |
| Keep angle | 1° |
| Axis | 2° |
| Gap | 2 mm |
| Row alignment | 2 mm |
| Clearance and boundary | 10 mm |
| Foreground threshold | 0.5 |

Gaps are measured on the rotated bounding rectangle, not on the chamfer or capsule silhouette. A count mismatch fails the scene immediately. `semantic_errors_v3` is in `src/knolling_model/arrangement/evaluation_v3.py`. It calls `semantic_errors` with the tolerances above. `evaluate_batch` in `src/knolling_model/evaluation.py` calls `geometry(..., 0.010, 0.010)`. The function itself is in `src/knolling_dataset/validator.py`.

These are prediction tolerances. A stored target is accepted under a different rule. Locked poses must match the source exactly, and `continuous_rows_v3` checks axis, gap, and row alignment at 2°, 2 mm, and 2 mm before the example is kept. The validator's own default, axis 5° and a 5.1 mm gap, is what the 100-scene pilot used. See [dataset-contract.md](dataset-contract.md).

This score is target-pose accuracy on synthetic RGB. It is not a robot-arm execution rate.

## Official result

The number below is the 640-scene holdout, scored once, at `v1.0.0` and `v2.0.0-rc1`. It is not the older 2,000-scene test, and it is not a validation split. Later commits on `main` did not repeat this measurement.

| System | Full | Partial | No-op | Overall |
| --- | ---: | ---: | ---: | ---: |
| v1.0.0 | 287/384 | 84/127 | 118/129 | 489/640 (76.4%) |
| v2.0.0 | 311/384 | 104/127 | 125/129 | 540/640 (84.4%) |

v2 kept every v1 success and added 51 (full 24, partial 20, no-op 7). Across 128 families the difference is +8.0 percentage points. A family bootstrap of 10,000 resamples, seed 20261112, gave a 95% interval of +4.8 to +11.6. Twenty-nine families improved and none got worse.

The holdout and the official 2,000-scene test are closed to training, threshold changes, and checkpoint selection. The command below repeats the sealed measurement.

```bash
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiments/end_to_end/score_holdout.py \
  --code-root . \
  --data artifacts/v2_independent_eval \
  --output /tmp/v2_holdout.jsonl
```

`score_holdout.py` refuses `generalization_20000` and any sample id that does not start with `v2hold_`. It aborts if the threshold is not 0.5.

## Demos

[The gallery](v2.0.0/demo-gallery/index.html) keeps ten validation scenes recorded at commit `26dca53`. Each motion figure draws the stored source pose and the stored predicted target pose. Seven pass and three fail. The README shows three passing full arrangements: that saved eight-object scene with vertical axes, a different eight-object scene with horizontal color rows in the right region, and a six-object scene. The last two were inferred on the validation split and accepted by the same judge. Neither set is a success rate. The earlier side-by-side renders are kept beside the gallery.
