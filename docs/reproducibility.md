# Reproducibility

## Environment

Python 3.12. Inference needs an NVIDIA GPU and the pins in `requirements-neural-lock.txt` (PyTorch 2.9.1, torchvision 0.24.1, transformers 4.57.3). CPU can import the packages. The adopted path sets its device to CUDA inside `scripts/run_relation_stable.py`.

```bash
python -m pip install -e ".[neural,test]"
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`HF_HUB_OFFLINE=1` requires the RT-DETR and MiniLM snapshots to be present in `HF_HOME`. The pinned revisions are in `configs/pretrained_rtdetr.json`:

- `PekingU/rtdetr_r18vd_coco_o365` at `386c43855d01bcd966e548f3296fbd2cd2a2a40b`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`

Git contains the code, configs, tests, and the demo gallery. It does not contain `artifacts/` or the Hugging Face cache. `*.pt` is gitignored, and the repository has no download URL for the weights. Place the files at the paths and SHA-256 values in [v2.0.0.md](v2.0.0.md) before loading the model. The arrangement checkpoint also needs `artifacts/planned_comparison/arrangement/shared/`. The neural-keep checkpoint that `load_current` reads needs `artifacts/checkpoint_store/`. The release table names the parent checkpoints. Those two blob directories have to sit beside the paths stored inside them.

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

The published system is reloaded with `load_adopted`. Tolerances for judging the returned pose are in [evaluation.md](evaluation.md).

## Training

The programs under `scripts/` and `experiments/` are the ones that produced the frozen checkpoints. Invoking `python scripts/run_*.py` is not a verified way to recreate the v2.0.0 weights. The generation scale and the smaller manifests used by individual stages are distinguished in [dataset-contract.md](dataset-contract.md).

Seeds fixed by the data code:

| Use | Seed |
| --- | --- |
| 20,000-scene generator | 120912 |
| Asset registry | 120913 |
| Independent holdout, recorded in that set's contract | 20261112 |
| Development-family splitter | `experiments/end_to_end/dev_splits.py` |
| Relation-training configs | 1203 |

The program that built the 128 holdout families is not in git.

## Validation

A direct check of the adopted decoder and split logic:

```bash
python -m pytest tests/neural/test_constraint_decode.py \
  tests/neural/test_geometry_decoder.py \
  tests/neural/test_partial_arrangement.py \
  tests/neural/test_inference.py \
  tests/neural/test_size_refinement.py \
  tests/neural/test_dev_splits.py
```

`python -m pytest` collects 641 tests. The recorded result is 547 passed, 0 failed, and 94 skipped. Each skip is an opt-in mark applied at collection time. With `RUN_RECOVERY_CUDA_TESTS=1`, the 83 CUDA tests passed on an RTX 4090.

Other opt-in results on the same machine:

- The RT-DETR pretrained forward test, the recovery CPU smoke test, the English-pilot audit, and the four Blender render tests passed. The render tests need `BLENDER_EXECUTABLE` pointed at Blender 5.1.0, plus `RUN_BLENDER_TESTS`, `RUN_BUILDER_TESTS`, and `RUN_MIGRATION_TESTS`.
- `tests/neural/test_pretrained.py` with `configs/pretrained.json` fails because `resnet18-f37072fd.pth` is absent from the torch hub cache. The test does not download the file.
- Three acceptance tests remain skipped. They require the legacy 100-scene pilot and the standalone v1.3 parent dataset, which are not present. The English pilot `artifacts/pilot_v1_3_en` does not satisfy those tests.

On validation family `expanded_09aa3f57968449ce8a09f291`, `load_adopted` returned a pose for every detected object under all three instructions. The official judge accepted the no-op scene. The full scene failed `vertical_gap`, and the partial scene failed `clearance`. That family is one pipeline check. It is not a success rate.

The 640-scene holdout was not part of these checks.
