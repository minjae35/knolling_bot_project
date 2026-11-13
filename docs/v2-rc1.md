# v2.0.0-rc1

The release candidate is the `v2.0.0-rc1` tag on `dev/knolling-v2`. Inference behavior matches `5efca4b` plus this record. `main` and `v1.0.0` stay at `26dca53`. Nothing in this record selects a new checkpoint or threshold.

## Frozen inference

Foreground threshold is `0.5`. Capacity is 8. The evaluator is unchanged: keep 1 mm and 1°, axis 2°, gap 2 mm, row 2 mm, clearance and boundary 10 mm. Geometry band `BAND_M` is 2 mm. Row repair and color separation each try at most 8 selector-ranked candidates. Duplicate detections inside 10 mm are dropped. `HELD_BELOW` stays 0.45.

Checkpoints, loaded by `load_adopted`:

| role | path | sha256 |
| --- | --- | --- |
| perception and arrangement base | `configs/structural_integration.json` | `e5ec3d0180ec5b5051d5a52cb4ed1286df5cbd694e54c4ebf1d5e76c58f302de` |
| joint row, order, and size | `artifacts/experiments/joint_stage/checkpoint-best.pt` | `02ed7db8d0122c4c5038b8870ea8bae833ad1effecc31b774062a20785d6159a` |
| selector | `artifacts/experiments/selector_full/checkpoint-best.pt` | `b21543e0b2a0764ceed262f25ad694330f93dc473ea5696ffced9046a67e2c6d` |
| selection head | `artifacts/experiments/selector_full/selection-head.pt` | `5345248902d662b56f7eaeaf9726b67adc24d2def64f52bf200f864f9d639024` |
| direction head | `artifacts/experiments/selector_full/direction-head.pt` | `97b37282fc9d2947ffb6175b6d16e5a31a320a713f548c3a511c4767bd384e38` |
| relation | `artifacts/phase1_relation_continue/checkpoint-best.pt` | `f70e1b65daae540ea26af1fc0b32a6a71fb46b42cdc10a9fa238e8475044dcbf` |

`/tmp/size_head.pt` is not loaded.

## Sealed holdout

`artifacts/v2_independent_eval` has 128 `v2hold_` families and 640 samples. It was not used for training, thresholds, or decoding choices. Scene families, RGB hashes, and asset families do not overlap the 20,000-scene train, validation, or test splits. Its objects come from the development asset pool, which that dataset never renders. Do not use this holdout, or the official test split, to change the model. Another development set and another final test are required for any further work.

Score v2 once:

```bash
python experiments/end_to_end/score_holdout.py \
  --code-root . \
  --data artifacts/v2_independent_eval \
  --output /tmp/v2_holdout.jsonl
```

Score v1.0.0 on the same files:

```bash
git worktree add /tmp/knolling-v1 v1.0.0
ln -sfn "$(pwd)/artifacts" /tmp/knolling-v1/artifacts
python experiments/end_to_end/score_holdout.py \
  --code-root /tmp/knolling-v1 \
  --data "$(pwd)/artifacts/v2_independent_eval" \
  --output /tmp/v1_holdout.jsonl
```
