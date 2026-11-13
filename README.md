# Knolling Bot

v2.0.0 predicts a target pose for every detected object from a top-down RGB
image and one English instruction. The published score is accuracy of that
pose on synthetic images. It is not a robot-arm execution rate.

| System | Full | Partial | No-op | Overall |
| --- | ---: | ---: | ---: | ---: |
| v1.0.0 | 287/384 | 84/127 | 118/129 | 489/640 (76.4%) |
| v2.0.0 | 311/384 | 104/127 | 125/129 | 540/640 (84.4%) |

The table is one sealed holdout of 128 scene families. v2 kept every v1
success and added 51. Details, tolerances, and the reproduction command are
in [docs/evaluation.md](docs/evaluation.md). Tag `v1.0.0` stays as the previous
baseline. Further training belongs on `dev/knolling-v3`.

## What the system does

Knolling here means putting loose objects into aligned, evenly spaced rows.
The input is one 576×384 overhead image and one sentence. The output is
`(x, y, θ)` for each detected object. No ground-truth pose is read at
inference time.

| Mode | Sentence | What moves |
| --- | --- | --- |
| Full | starts with `Arrange all objects` | every object |
| Partial | `Arrange only the red objects` (or green, or blue) | that color |
| No-op | `Keep every object at its current position and orientation.` | nothing |

The sentence also names the region, the gap (15, 20, or 30 mm), color
grouping, size order, and the long-axis direction. The pipeline is in
[docs/architecture.md](docs/architecture.md).

```text
RGB ──► RT-DETR ──► objects
text ──► goal vector ──► selector ──► constrained rows ──► geometry decoder
partial / no-op keep the objects the sentence does not ask to move
```

## Quick start

Python 3.12 and a CUDA GPU. Pins are in `requirements-neural-lock.txt`.

```bash
python -m pip install -e ".[neural,test]"
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Weights and datasets are not in git. Put them at the paths in
[docs/v2.0.0.md](docs/v2.0.0.md), then call `load_adopted` and
`poses_from_detection` as shown in
[docs/reproducibility.md](docs/reproducibility.md). An external copy of those
files was started and is not verified, so this repository alone cannot reload
the model yet.

## Layout

| Path | Contents |
| --- | --- |
| `src/knolling_dataset/` | scenes, English, teacher, renderer, validator |
| `src/knolling_model/` | detector, arrangement, adopted inference |
| `experiments/end_to_end/` | loader, holdout score, full-only demo |
| `configs/` | pins for the weights the loader still opens |
| `scripts/` | training runners; not a verified retrain recipe |
| `tests/` | unit and regression tests |
| `docs/` | architecture, data, evaluation, reproducibility |
| `archive/` | earlier experiment notes, kept on purpose |

## Demos

Ten scenes, seven passes and three failures, captured at `26dca53`. Not a
success rate.

[Gallery](docs/v2.0.0/demo-gallery/index.html)

![Full arrangement](docs/v2.0.0/demo-gallery/01_full_pass.png)

## Further reading

- [Architecture](docs/architecture.md)
- [Dataset](docs/dataset.md)
- [Evaluation](docs/evaluation.md)
- [Reproducibility](docs/reproducibility.md)
- [Limitations](docs/limitations.md)
- [Third-party notices](docs/third-party-notices.md)

There is no project `LICENSE` yet. Do not assume the code is open source.
