# Dataset

The published training distribution is
`artifacts/generalization_20000/dataset`.

| | |
| --- | --- |
| Families | 4,000 |
| Samples | 20,000 |
| Train / validation / test | 16,000 / 2,000 / 2,000 |
| Scene seed | 120912 |
| Asset-registry seed | 120913 |
| Schema | `expanded_knolling_20000_v1` |
| Teacher | `continuous_rows_v3` |

Each family is one RGB image and five instructions that share it: horizontal
full, vertical-axis full, one-color partial, keep-all no-op, and one
contrasting full instruction. Object counts run from 2 to 8. Shapes are box,
chamfer, and capsule. Colors are red, green, and blue. Images are 576×384.

The split is fixed before rendering. Family index `i` uses
`bucket = (i // 20) % 10`: buckets 0–7 are train, 8 is validation, and 9 is
test. Code: `partition` in `src/knolling_model/experiments/expanded_data.py`.

Assets are procedural. The schema field `license` defaults to `CC0-1.0` for
those generated shapes. That field is not a license for this repository.
Geometry families do not cross the train, validation, and test pools.

## Generation

`expanded_data.generate(root)` builds the 20,000-scene tree. It calls the
dataset package for scenes, the English renderer, the teacher, and Blender.
This cleanup did not regenerate the set. Treat a fresh run as unverified until
its `annotations.json` hash matches the hash in [v2.0.0.md](v2.0.0.md).

Blender is an external program (this release used 5.1.0, Cycles, CPU, 32
samples). It is not vendored. Cross-machine PNG bytes are not guaranteed.

The geometry contract, regions, and teacher search are in
[dataset-contract.md](dataset-contract.md). Sentences at the end of that note
that describe a 100-family pilot, or say that no training has been run, belong
to that earlier pilot. They do not describe the 20,000-scene set. English
wording conventions are in
[english-instruction-defaults.md](english-instruction-defaults.md).

## Official holdout

`artifacts/v2_independent_eval` is a later set: 128 families, 640 samples,
schema `v2_independent_eval_v1`, scene seed 20261112, the same registry seed
120913, development assets only. The contract stored with that set says it was
scored once and is not for model selection.

The script that built those 128 families is not in this repository. The set
can be scored with the command in [evaluation.md](evaluation.md). It cannot be
regenerated from the git tree alone.

Both datasets are gitignored. SHA-256 for the annotation files is in
[v2.0.0.md](v2.0.0.md).
