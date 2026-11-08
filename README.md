# Knolling Bot

Develop on **`main`**. The current system combines the independently trained
RT-DETR Perception model with v4 Goal Encoder → split XY/Orientation → Neural
Keep Arrangement. Inference uses BEV RGB and natural language only. Keep routing
uses a learned prediction; there is no GT mask or rule-based planner in forward.

## Current models and limitations

`configs/current_models.json` pins both Best checkpoint paths and SHA-256 hashes.
Binaries and datasets are local, ignored assets, not Git downloads.

| Component | Preserved Best | Verified scope |
|---|---|---|
| Arrangement Neural Keep | Validity **40/76**, Counterfactual **7/35** | 76 training scenes, GT source objects |
| Independent RT-DETR Perception | Count **4/10**, mAP50:95 **0.945066** | 10 held-out source families |

The accepted Arrangement checkpoint is update 400 from the single Neural Keep
run. The failed Pairwise model is excluded. Perception still has extra
predictions and size/orientation/semantic errors. These individual results do
**not** establish End-to-End task success or Arrangement generalization.
The adapter rejects more than 8 detections explicitly; it never truncates to
GT count or silently changes the confidence threshold.

## Read-only verification

Run from this repository with the existing `knolling` environment and cached
pretrained weights:

```sh
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n knolling python -m pytest -q
conda run --no-capture-output -n knolling python scripts/verify_current_models.py \
  --device cuda --output artifacts/verification/new-unique-run
```

The second command performs **zero optimizer steps**. It verifies both pinned
checkpoint hashes, both loaders/fingerprints, Arrangement's 76-scene replay,
Perception's 10-source Validation replay, and actual RGB→predicted objects→Goal
Encoder→Neural Keep forward. Use a fresh output directory each time.

For application code, `knolling_model.current.load_current()` returns the
composed read-only system. Call `predict(images, commands, frame_ids)` with RGB
float tensors `[B,3,384,576]` in `[0,1]`. Outputs preserve detector query IDs,
physical metres, pi-periodic angles, predicted object provenance and final
Target Poses. No targets or annotations are accepted by that method.

## Maintained code

- `src/knolling_model/perception/`: source-only dataset, RT-DETR, physical
  representation, checkpoint/provenance, metrics and Arrangement adapter.
- `src/knolling_model/current.py`: pinned Best loaders and composition.
- `src/knolling_model/recovery/`: Goal Encoder, split pose/Keep, validated
  dataset/teacher contracts, evaluation, lightweight checkpoint/resume support.
- `src/knolling_dataset/`: common source/target records, renderer, teacher and
  data-integrity tooling shared by training/evaluation.
- `scripts/run_relative_task.py`, `scripts/run_perception.py`: existing bounded
  training entry points; training still requires a separate explicit request.

Keep architecture constructors/readers and their regression tests needed to
restore historical checkpoints. Historical one-off scripts are isolated under
`archive/experiments/`; AR/v3 are compatibility implementations, not alternative
current models. Historical configuration files referenced by saved checkpoints
and safety tests remain available. The previous S0 guide is preserved in
[docs/legacy-s0-readme.md](docs/legacy-s0-readme.md).

## Records and local assets

- [Repository consolidation and preservation](docs/repository-consolidation.md)
- [Neural Keep evidence and results](docs/recovery-v4-neural-keep.md)
- [Perception GPU result and limitations](docs/perception-gpu-results.md)
- [Checkpoint storage contract](docs/recovery-checkpoint-storage.md)
- [Git identity policy](docs/git-identity.md)

Perception artifacts now reside at `artifacts/perception/`; Arrangement runs,
datasets and shared checkpoint blobs retain their original paths. Do not delete
shared blobs, anchors, journals or experiment folders without dependency checks.
Git history/branch inventory is backed up locally under
`artifacts/repository_consolidation/20261108/`. No large checkpoint is added to Git.
