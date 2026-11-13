# Reproducibility

Python 3.12. Inference needs an NVIDIA GPU and the pins in
`requirements-neural-lock.txt` (PyTorch 2.9.1, torchvision 0.24.1,
transformers 4.57.3). CPU can import the packages. The adopted path sets its
device to CUDA inside `scripts/run_relation_stable.py`.

```bash
python -m pip install -e ".[neural,test]"
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`HF_HUB_OFFLINE=1` means the RT-DETR and MiniLM snapshots must already be in
`HF_HOME`. The pinned revisions are in `configs/pretrained_rtdetr.json`:

- `PekingU/rtdetr_r18vd_coco_o365` at `386c43855d01bcd966e548f3296fbd2cd2a2a40b`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`

## What a clone contains

Git has the code, configs, tests, and the demo gallery. It does not have
`artifacts/` or the Hugging Face cache. `*.pt` is gitignored. There is no
download URL in this repository. An external copy of those files was started
and has not been verified here, so this note does not say that the backup is
complete.

Place the files at the paths and SHA-256 values in [v2.0.0.md](v2.0.0.md)
before loading the model. The arrangement checkpoint also needs
`artifacts/planned_comparison/arrangement/shared/`. The neural-keep checkpoint
that `load_current` reads needs `artifacts/checkpoint_store/`. Both are
recorded in that release note's weight table only by their parent checkpoints.
Those two blob directories have to sit next to the paths the checkpoints store.

## Inference

From the repository root, with `src`, `scripts`, and `experiments/end_to_end`
on `PYTHONPATH`:

```python
import run_phase1_pairwise_affine as runner
from knolling_model.experiments.size_refinement import poses_from_detection
from score_end_to_end import load_adopted

runner.setup()
system, relation, model, head, _saved = load_adopted()
output, objects, visual = poses_from_detection(
    system, relation, model, images, texts, head)
```

`output['xy']` and `output['angle']` are the target pose. `texts` is one
English sentence per image.

| Sentence | Mode |
| --- | --- |
| starts with `Arrange all objects` | full |
| `Arrange only the red objects` (or green, or blue) | partial |
| `Keep every object at its current position and orientation.` | no-op |

A one-scene smoke of all three modes is part of the check for this branch.
The 640-scene table was not recomputed.

## Training

The scripts in `scripts/` and `experiments/` are the programs that produced
the frozen checkpoints. This cleanup did not retrain them. Do not treat a
`python scripts/run_*.py` invocation as a verified recipe for reproducing
v2.0.0. The verified way to reload the published system is `load_adopted`.

Seeds that the data code does fix:

| Use | Seed |
| --- | --- |
| 20,000-scene generator | 120912 |
| Asset registry | 120913 |
| Independent holdout, recorded in that set's contract | 20261112 |
| Development-family splitter | `experiments/end_to_end/dev_splits.py` |

The holdout generator itself is not in git. Relation-training configs use seed
1203. That is the training script's seed, not a seed this cleanup re-ran.

## Tests

```bash
python -m pytest tests/neural/test_constraint_decode.py \
  tests/neural/test_geometry_decoder.py \
  tests/neural/test_partial_arrangement.py \
  tests/neural/test_inference.py \
  tests/neural/test_size_refinement.py \
  tests/neural/test_dev_splits.py
```

Some files under `tests/neural/` still target the older recovery stack and
were already failing to collect before this cleanup. Those failures are listed
in the branch report. They are not the v2 regression set above.
