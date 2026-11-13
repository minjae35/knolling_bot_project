# Single Perception GPU run — 2026-11-08

Exactly **one production GPU training run** completed: **500 optimizer updates
in 122.44 seconds**, with Validation-selected Best at update **500**. Detection
improved substantially, but the current checkpoint does **not** satisfy the
existing source-accuracy requirements for a reliable End-to-End Demo.

## Fixed execution and preservation

The existing RT-DETR-R18, physical Pose heads, losses, confidence threshold 0.5,
Dataset Loader and Source-family split were reused without architecture or
hyperparameter changes. Train is 70 distinct RGB sources / 341 objects;
Validation is 10 / 56. Reserved Test and Development were not evaluated.

The precommitted single-run configuration is `configs/perception_gpu_once.json`:
seed 1203, FP32, batch 2, AdamW backbone LR 1e-5 / other LR 1e-4, weight decay
1e-4, clipping 1, maximum 500 updates / 1,800 seconds, Validation every 50 updates,
120-second final reserve. This is the original fixed proposal, not a search.
The run reached the update limit; no continuation or automatic retraining ran.

Execution used `/venv/knolling/bin/python`, the RTX 4090, `tmux` session
`knolling-perception-once`, and live local `tqdm`. GPU process queries before
CUDA preflight and production launch found no other compute processes. No
other Agent's training processes were active. The run receipt binds the config,
CUDA report and code commit `a957a49` to the user's single-run authorization.

The original pinned pretrained safetensors SHA-256 was verified:
`eadade7a507b87cabe5d6a75b935dbd9fc59b1b843f36f7e3b24eafc856c0774`.
Missing/unexpected/mismatched/error loading lists were all empty. The existing
foreground/physical heads were freshly initialized as designed; no Arrangement
weights or old trained model were substituted for the pretrained initialization.

Updated runtime instrumentation passed CPU preflight and **30 tests** before
launch. The actual CUDA preflight passed native-forward parity (box error 0),
finite nonzero detection/physical gradients, and unchanged weights/buffers.
It performed **zero optimizer updates** and peaked at **1,259.4 MiB** allocated.
The production history contains exactly updates 1 through 500, once each.

Arrangement code/checkpoints and existing dataset files were preserved. The
managed `/opt/knolling-notify/notify.py` dispatcher delivered the registered
terminal completion event; no progress worker or direct transport was added.

## Best Validation performance

All errors below use the **56 actual confident IoU-matched detections**; these
are not Oracle/GT-assignment errors. All 56 annotated objects were found, with
9 extra false positives. AP uses the existing documented single-class physical
envelope evaluator: 300 queries/image, 101 recall points, no area/crowd split.

| Measurement | Best Validation |
| --- | ---: |
| Exact Object Count Accuracy | **4/10 = 40%** |
| AP50 (0–1) | **0.995342** |
| mAP50:95 (0–1) | **0.945066** |
| TP / FP / FN | **56 / 9 / 0** |
| Precision / Recall | **86.15% / 100%** |
| Position error mean / P95 | **0.442 / 0.998 mm** |
| Major size error mean / P95 | **1.923 / 5.190 mm** |
| Minor size error mean / P95 | **3.202 / 7.362 mm** |
| Per-object max size-component error mean / P95 | **3.658 / 7.436 mm** |
| Pi-periodic orientation error mean / P95 | **3.075 / 7.563 degrees** |
| Color / shape accuracy on matches | **39.29% / 67.86%** |

High AP and imperfect count accuracy are compatible: AP evaluates ranked
predictions, whereas count accuracy includes every prediction above the fixed
0.5 operating threshold. No threshold tuning, NMS addition, GT count filtering,
or post-training calibration was applied to improve the reported numbers.

A forward-only Train assessment found Count Accuracy **39/70 = 55.71%**,
mAP50:95 **0.961398**, TP/FP/FN **341/40/0**. Extra detections and physical/
semantic accuracy remain problems even on training views; held-out scene
count alone does not explain the limitation.

## Best checkpoint and replay

Best:
`/root/.codex/worktrees/perception/knolling_bot_project/artifacts/perception/gpu-once-20261108/training/best.pt`

SHA-256:
`c02dc5e9501a39cfcb9d1de8d512ace765050d675155e66bbde0ec2889105491`

`last.pt` is retained separately, SHA-256
`d41bbbbf0564238ce027cb3f6930d790824f2223f5aaed2ef40f23540ce477ab`.
Both contain only independent Perception weights plus optimizer/RNG/config/data
provenance. The binaries remain local and ignored by Git; source and compact
evidence are committed/pushed. The dataset fingerprint is unchanged:
`668ec630a9d189d1c44041a44fc1bfff063b0b61ff69249551e9bb7d38cde519`.

Best was reloaded inside training and again in a separate read-only CUDA
assessment. All Validation metrics exactly reproduced. The separate assessment
made **zero optimizer updates**. No second training run occurred.

RGB-only exported inference is available with the unchanged CLI:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src HF_HUB_OFFLINE=1 \
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
/venv/knolling/bin/python scripts/run_perception.py infer \
  --checkpoint artifacts/perception/gpu-once-20261108/training/best.pt \
  --image /absolute/path/to/calibrated-576x384-bev-rgb.png \
  --output /tmp/predicted-objects.json
```

## Adapter and Demo decision

The actual predicted JSON from all 80 Train/Validation sources passed the
independent object schema and the Arrangement adapter without Oracle input.
Positions, dimensions, angles, IDs and object counts were preserved; the
existing object feature dimension is 12. No Arrangement forward/checkpoint was
used or modified.

The existing optional 8-object capacity rejected **2/10 Validation** sources
with 10 predicted objects, and **8/70 Train** sources with 9–11 predictions.
The adapter preserves those detections in the independent representation and
raises on the explicit legacy capacity; it never silently truncates to GT count.

**Decision: suitable as a trained detection prototype, not yet a reliable
End-to-End Demo input.** The existing source precision gate requires correct
count/coverage, no FP/FN, position and max-component size P95 <=1 mm, angle P95
<=2 degrees, and correct color/shape. Position meets the P95 threshold here;
extra detections, size, orientation and semantics do not. The additional strict
gate is a reference from `recovery/stages.py`, not a new hyperparameter chosen
after observing the run. End-to-end task success was not evaluated.

Remaining problems are duplicate/extra detections at the fixed threshold,
physical dimension/orientation accuracy, low color/shape accuracy needed for
language-based selection, and capacity overflow caused by the extra detections.
Further training changes or integration require a separate future task; this
run was not extended or repeated.

## Saved evidence

- [CUDA loading/forward/gradient report](perception-gpu-evidence/cuda-preflight.json)
- [Single-run launch receipt](perception-gpu-evidence/launch-receipt.json)
- [Training report and Best Validation metrics](perception-gpu-evidence/training-report.json)
- [Complete Best/adapter/Demo assessment](perception-gpu-evidence/demo-assessment.json)
- [Validation curve at every fixed evaluation](perception-gpu-evidence/validation-curve.json)

Complete predictions and console/history logs remain under
`artifacts/perception/gpu-once-demo-evaluation/` and
`artifacts/perception/gpu-once-20261108/` in this worktree.
