# Dataset

The training distribution is `artifacts/generalization_20000/dataset`.

| | |
| --- | --- |
| Families | 4,000 |
| Samples | 20,000 |
| Train / validation / test | 16,000 / 2,000 / 2,000 |
| Scene seed | 120912 |
| Asset-registry seed | 120913 |
| Schema | `expanded_knolling_20000_v1` |
| Teacher | `continuous_rows_v3` |
| Instruction contract | `english_rows_v1` |

Each family is one RGB image and five instructions that share it: horizontal full, vertical-axis full, one-color partial, keep-all no-op, and one contrasting full instruction. Object counts run from 2 to 8. Shapes are box, chamfer, and capsule. Colors are red, green, and blue. Images are 576 × 384.

The split is fixed before rendering. Family index `i` uses `bucket = (i // 20) % 10`: buckets 0–7 are train, 8 is validation, and 9 is test. The function is `partition` in `src/knolling_model/experiments/expanded_data.py`. Geometry families do not cross the train, validation, and test pools.

Assets are procedural. The schema field `license` defaults to `CC0-1.0` for those generated shapes. That field is not a license for this repository.

## Generation

`expanded_data.generate(root)` builds the tree. It calls the dataset package for scenes, `english_instruction` for the sentence, `continuous_rows_v3` for the target, and Blender for the RGB image. A new run is the same set only when `annotations.json` matches the SHA-256 in [v2.0.0.md](v2.0.0.md).

Blender is an external program. This release used 5.1.0, Cycles, CPU, and 32 samples. It is not vendored. Cross-machine PNG bytes are not guaranteed.

Workspace geometry, the difference between this teacher and the earlier 100-scene pilot, and the difference between dataset acceptance and prediction scoring are in [dataset-contract.md](dataset-contract.md). Sentence conventions for the v1.3 balanced labels are in [english-instruction-defaults.md](english-instruction-defaults.md). Those balanced labels are not the teacher of this 20,000-scene set.

## Official holdout

`artifacts/v2_independent_eval` is a later set: 128 families, 640 samples, schema `v2_independent_eval_v1`, scene seed 20261112, registry seed 120913, and development assets only. Its contract says the set was scored once and is not for model selection.

The script that built those 128 families is not in this repository. The set can be scored with the command in [evaluation.md](evaluation.md). It cannot be regenerated from the git tree alone.

Both datasets are gitignored. SHA-256 values for the annotation files are in [v2.0.0.md](v2.0.0.md). The paths above are the paths the code opens.
