# Independent RGB Perception

The `codex/perception` branch adds `knolling_model.perception` without modifying
Arrangement, its branches, source files, checkpoints, or training processes.
It produces physical object observations from calibrated BEV RGB. Final
end-to-end coupling and joint training remain deferred until Arrangement is stable.

## Reused implementation

`ObjectPerception` reuses the existing `RecoveryPerception`: pinned pretrained
RT-DETR-R18, foreground detection, deterministic deformable sampling, fixed BN
statistics, isolated training-only denoising, native VFL/L1/GIoU detection
losses, and supervised physical major/minor dimensions and pi-periodic angle
heads. No new detector, pose solver, rule-based planner, text encoder, or
Arrangement model is introduced. Native top-300 proposals are model capacity;
inference never uses GT count or GT object attributes to select predictions.

`forward(images)` and `predict(images, frame_ids)` accept RGB only. Training's
`supervised(images, truths)` accepts source labels for denoising and is rejected
in evaluation mode. Existing tests verify normal queries cannot read GT
denoising tokens. Physical dimensions are learned separately from the
axis-aligned detector boxes; rotated box widths are not intrinsic dimensions.

## Verified data and split

The immutable data remains at `/workspace/knolling_bot_project/artifacts/` and
is read from the worktree through explicit configuration paths.

| Data | Annotation records | Unique RGB source families |
| --- | ---: | ---: |
| Pilot English | 130 | 100 |
| Recovery v3 | 76 | 25, all already in Pilot Train |
| Perception Train | 89 Pilot records | 70, 341 objects |
| Perception Validation | 13 Pilot records | 10, 56 objects |
| Reserved Test | 15 Pilot records | 10, 44 objects |
| Reserved Development | 13 Pilot records | 10, 54 objects |

Recovery contains source poses, dimensions, semantics, and RGB references, but
does not duplicate Pilot's object-level mask/BBox fields. The loader requires
exact source/split/RGB equality with a hash-verified Pilot annotation. It creates
no missing annotation, targets, pseudo-labels, or new examples. Recovery does
not increase the amount of visual training data.

`SourceCatalog` checks all annotation/RGB/mask/render hashes, calibration,
instance-to-object mapping, stored mask boxes (exclusive maximum pixel bounds),
source center projections, physical dimensions, envelope annotations, and
independent RGB/mask/pose reprojection using existing `validate_render`.
Missing or inconsistent labels fail immediately. Masks are used for audit;
this implementation does not train a segmentation head.

The original source-family Train/Validation/Test/Development assignments are
preserved. Multiple commands for a source yield one perception sample. Source
family, identical RGB/state, and asset-family checks reject cross-partition
leakage. Neither a Teacher replay nor target layout is needed by the loader.

## Public object representation

`ObjectFrame.to_dict()` emits the versioned `knolling_objects_v1` JSON contract.
The contract is established from `docs/dataset-contract.md`,
`knolling_dataset/schema.py`, `knolling_model/coordinates.py`, and the actual
`recovery.contracts.Objects` consumed by the current Arrangement.

| Field | Definition |
| --- | --- |
| `frame_id` | Caller-provided image/frame identifier |
| `source_mode` | `predicted_rgb` or explicitly constructed `oracle_gt` |
| `object_id` | `query_NNN` in predictions; frame-local, not a persistent track or GT identity |
| `query_index` | Original detector query row; never GT-matched during inference |
| `xy_m` | Workspace center in metres; bottom-left origin, +x right, +y up |
| `size_ab_m` | Intrinsic major/minor lengths in metres, a >= b > 0 |
| `theta_rad` | Major-axis CCW angle from +x; period pi, canonical [-pi/2, pi/2) |
| `confidence` | Foreground sigmoid score; fixed default threshold 0.5 |
| `box_xyxy_image` | Normalized image y-down physical-envelope AABB; distinct from mask silhouette box |
| `color_probability` | Red, green, blue probabilities, in that order |
| `shape_probability` | Box, chamfer, capsule probabilities, in that order |

Workspace is 0.600 x 0.400 m; input is RGB `[B,3,384,576]` in [0,1]. Pixel
centers obey `px=576*x/0.6-0.5`, `py=384*(1-y/0.4)-0.5`. The inherited
detector performs its reviewed internal 640x640 resize. No uncalibrated-image
fallback or automatic workspace assumption is added. Height/depth/roll/pitch
are not predicted; current physical supervision and Arrangement use planar
major/minor dimensions and orientation.

`adapter.to_arrangement` alone imports the existing padded `Objects` interface.
It returns that interface and an ID/confidence/provenance sidecar. It preserves
metres and orientation, pads empty detections with one **invalid** slot, and
raises on an explicitly requested capacity overflow. It never truncates to GT
count or fabricates detections. `oracle_frame` is a separate diagnostic helper;
passing it to the default predicted adapter raises a provenance error.

The model and checkpoints contain no Arrangement weights. Checkpoint inference
compatibility hashes only the perception graph and reused physical-head
definitions; changing an Arrangement class or this adapter does not invalidate
trained perception weights. Full source hashes are also retained as audit
metadata. This JSON bridge is an inference/export boundary, not a joint-gradient
API; joint training is intentionally deferred.

## Commands and verification

Use the existing Python 3.12 environment `/venv/knolling/bin/python` with the
pinned neural requirements. Run commands from this worktree/repository root.
All assets must already exist in the external Hugging Face cache; the loader
verifies the pinned detector's SHA-256 and does not download new models.

```sh
export PYTHONPATH=src
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=''

/venv/knolling/bin/python scripts/run_perception.py audit \
  --output /tmp/perception-annotation-audit.json

# Requires a new output directory. Exactly one disposable CPU optimizer update.
/venv/knolling/bin/python scripts/run_perception.py cpu-check \
  --output artifacts/perception/NEW_CPU_PREFLIGHT

/venv/knolling/bin/python scripts/run_perception.py infer \
  --checkpoint artifacts/perception/cpu-final/initial.pt \
  --image /workspace/knolling_bot_project/artifacts/pilot_v1_3_en/scenes/sf_38312dd8424b45d37d66/source/rgb.png \
  --output /tmp/perception-frame.json

/venv/knolling/bin/python scripts/run_perception.py evaluate \
  --checkpoint artifacts/perception/cpu-final/initial.pt \
  --split validation --output /tmp/perception-validation.json

/venv/knolling/bin/python -m pytest -q \
  tests/neural/test_standalone_perception.py \
  tests/neural/test_recovery_perception.py \
  tests/neural/test_recovery_contracts.py tests/neural/test_rtdetr.py
```

The CPU preflight traverses all 100 sources, evaluates all 10 Validation
families, and performs Forward/Backward on actual 2-object and 8-object Train
sources. It checks finite nonzero gradients in backbone, encoder, proposals,
decoder, foreground, physical heads, and denoising embedding, then verifies
exact predictions after strict checkpoint restoration and adapter roundtrips.
The disposable update is not a trained model or a new initialization for training.

Checkpoints store only perception state, optimizer/RNG state where applicable,
data/config/model provenance, update count, time, purpose, and metrics. Atomic
replacement plus SHA-256 sidecars detects incomplete/corrupt writes. No resume
or automatic retraining CLI is provided. Tests additionally reproduce the next
optimizer update after restoring optimizer and RNG state.

## Evaluation and current limitations

Metrics include exact count accuracy/MAE; TP/FP/FN, precision/recall at fixed
confidence 0.5 and envelope IoU >=0.5; AP50, AP75 and mean AP over IoU .50:.05:.95.
AP uses 101 recall points and all 300 detector queries, one foreground class,
without area/crowd stratification. This is a documented class-agnostic evaluator,
not the official COCO evaluator. Operational pose errors use actual confident
IoU-matched detections, with position in mm, separate major/minor errors in mm,
and pi-periodic angle error in degrees. Coverage is always reported; no matches
means **null**, not zero error. A separately labeled GT-assisted assignment
diagnostic is not operational accuracy or a checkpoint selection criterion.

The initial foreground and physical heads are newly initialized. Their
Validation results are baseline measurements, not evidence of trained accuracy.
See `perception-cpu-results.md` and committed JSON evidence. Test and Development
were inspected for annotation integrity only; no prediction evaluation, tuning,
or optimization used them.

## Single-run training boundary

The initial implementation task was **CPU verification only, no exclusive GPU time**.
The user subsequently authorized exactly one GPU run; its completed results and
remaining Demo blockers are in [GPU results](perception-gpu-results.md).
`configs/perception_cpu.json` retains `allow_training=false`, CPU device,
no GPU allocation and no preflight authorization reference. The separate
`perception_gpu_once.json` records the one completed authorized GPU run. Joint
training, search, automatic retraining and Telegram progress workers remain absent.

The training path requires an explicitly confirmed allocation. The executed
fixed budget was FP32, seed 1203, micro/effective batch 2,
AdamW backbone LR 1e-5, other LR 1e-4, weight decay 1e-4, norm clip 1, at most
500 updates and 1,800 seconds including setup and validation. A 120-second final
reserve stops new updates early to save/reload/evaluate. Deadlines are checked
between operations; no in-flight optimizer/kernel is forcibly interrupted.
Actual CUDA loading, forward, gradients, training and checkpoint replay passed;
see the GPU report for measured accuracy and remaining limitations.

A future authorized config must explicitly enable training, select CUDA, refer
to a passing CPU preflight checkpoint with the same data/training code, and
provide `gpu_exclusive_window` with `device_uuid`, `start_unix`, `end_unix`, and
the allocation's `authorization`. `CUDA_VISIBLE_DEVICES` must equal that UUID.
The guard rejects occupied GPUs and windows that cannot cover the budget.
Launching `train` under the default CPU config fails before creating a run or
allocating CUDA. Do not infer allocation from momentary zero GPU utilization.

Training visits each Train source once per shuffled epoch, selects best only
by Validation mAP/count/recall/position, retains the initial candidate, saves
independent best/last checkpoints, and verifies best's Validation replay.
CPU tests exercise the bounded orchestration and best preservation with an
explicit test double. The separate completed GPU run provides direct execution
evidence; it does not meet the existing source-accuracy gate for an End-to-End Demo.

Remaining work: sufficient object-count, physical and semantic accuracy,
broader visual/asset coverage (only 70 Train sources), real-camera calibration/
domain validation, and a future adapter to
the stabilized Arrangement architecture. End-to-end and joint accuracy remain
unmeasured.
