# Reproducibility

Python 3.12. Inference needs an NVIDIA GPU and the pins in `requirements-neural-lock.txt` (PyTorch 2.9.1, torchvision 0.24.1, transformers 4.57.3). CPU can import the packages. The adopted path sets its device to CUDA inside `scripts/run_relation_stable.py`.

```bash
python -m pip install -e ".[neural,test]"
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`HF_HUB_OFFLINE=1` means the RT-DETR and MiniLM snapshots must already be in `HF_HOME`. The pinned revisions are in `configs/pretrained_rtdetr.json`:

- `PekingU/rtdetr_r18vd_coco_o365` at `386c43855d01bcd966e548f3296fbd2cd2a2a40b`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`

## What a clone contains

Git has the code, configs, tests, and the demo gallery. It does not have `artifacts/` or the Hugging Face cache. `*.pt` is gitignored. There is no download URL in this repository. This note does not record a checksum-verified copy of the weights outside the machine that produced them.

Place the files at the paths and SHA-256 values in [v2.0.0.md](v2.0.0.md) before loading the model. The arrangement checkpoint also needs `artifacts/planned_comparison/arrangement/shared/`. The neural-keep checkpoint that `load_current` reads needs `artifacts/checkpoint_store/`. Both are recorded in that release note's weight table only by their parent checkpoints. Those two blob directories have to sit next to the paths the checkpoints store.

## Inference

From the repository root, with `src`, `scripts`, and `experiments/end_to_end` on `PYTHONPATH`:

```python
import run_phase1_pairwise_affine as runner
from knolling_model.experiments.size_refinement import poses_from_detection
from score_end_to_end import load_adopted

runner.setup()
system, relation, model, head, _saved = load_adopted()
output, objects, visual = poses_from_detection(
    system, relation, model, images, texts, head)
```

`output['xy']` and `output['angle']` are the target pose. `texts` is one English sentence per image.

| Sentence | Mode |
| --- | --- |
| starts with `Arrange all objects` | full |
| `Arrange only the red objects` (or green, or blue) | partial |
| `Keep every object at its current position and orientation.` | no-op |

On validation family `expanded_09aa3f57968449ce8a09f291`, this entry point returned a pose for every detected object under all three instructions. The official judge accepted the no-op scene. The full scene failed `vertical_gap`, and the partial scene failed `clearance`, under the tolerances in [evaluation.md](evaluation.md). That family is one check of the pipeline. It is not a success rate, and it is not the 640-scene holdout.

## Training

The scripts in `scripts/` and `experiments/` are the programs that produced the frozen checkpoints. They were not re-run for the release note or for later documentation commits. A `python scripts/run_*.py` invocation is not a verified recipe for reproducing v2.0.0. The verified way to reload the published system is `load_adopted`.

Seeds that the data code does fix:

| Use | Seed |
| --- | --- |
| 20,000-scene generator | 120912 |
| Asset registry | 120913 |
| Independent holdout, recorded in that set's contract | 20261112 |
| Development-family splitter | `experiments/end_to_end/dev_splits.py` |

The holdout generator itself is not in git. Relation-training configs use seed 1203. That is the training script's seed. It was not re-run for this tree.

## Tests

A fast check of the adopted path:

```bash
python -m pytest tests/neural/test_constraint_decode.py \
  tests/neural/test_geometry_decoder.py \
  tests/neural/test_partial_arrangement.py \
  tests/neural/test_inference.py \
  tests/neural/test_size_refinement.py \
  tests/neural/test_dev_splits.py
```

On the current `main` checkout, `python -m pytest` collected 641 tests and reported 547 passed, 0 failed, and 94 skipped. Every skip is an opt-in mark evaluated at collection time. With `RUN_RECOVERY_CUDA_TESTS=1`, the 83 CUDA tests passed on the RTX 4090 in this instance.

Further opt-in runs on the same machine:

- The RT-DETR pretrained forward test, the recovery CPU smoke test, the English-pilot audit, and the four Blender render tests passed. The render tests need `BLENDER_EXECUTABLE` pointed at Blender 5.1.0 and `RUN_BLENDER_TESTS`, `RUN_BUILDER_TESTS`, and `RUN_MIGRATION_TESTS` set.
- `tests/neural/test_pretrained.py` with `configs/pretrained.json` fails because `resnet18-f37072fd.pth` is not in the torch hub cache. The test refuses to download it.
- Three acceptance tests stay skipped. They require the legacy 100-scene pilot and the standalone v1.3 parent dataset, and those trees are not on this machine. The English pilot `artifacts/pilot_v1_3_en` is a different deliverable and does not satisfy those tests.

The 640-scene holdout was not part of this test run.
