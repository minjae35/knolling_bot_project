# RT-DETR-R18 integration contract

The user approved replacing the randomly initialized Object Decoder with
pretrained RT-DETR-R18 on 2026-11-03. The legacy backend, configs, dataset,
Teacher, GT, evaluator and old run directories remain available. This is an
internal neural perception module, not an external layout planner.

## Pretrained assets and implementation

- Hugging Face `PekingU/rtdetr_r18vd_coco_o365`, revision
  `386c43855d01bcd966e548f3296fbd2cd2a2a40b`.
- `model.safetensors` SHA-256:
  `eadade7a507b87cabe5d6a75b935dbd9fc59b1b843f36f7e3b24eafc856c0774`.
- This is the complete Objects365-pretrained, COCO-finetuned detector, not an
  ImageNet-only R18 backbone: backbone, hybrid encoder, proposal score/box
  heads, query-position head, all three decoder layers, and all three
  classifier/box heads load from the checkpoint without missing keys.
- Use the pinned `transformers==4.57.3`, torch 2.9.1+cu128 environment. The
  adapter deliberately rejects an unreviewed Transformers version/topology.
- Loader verifies the weight hash, checkpoint preprocessing and complete
  loading. Downloads go to the ignored external Hugging Face cache, never Git.

The [official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR) identifies
the COCO+Objects365 recipe as COCO fine-tuning from Objects365 weights. The
[checkpoint](https://huggingface.co/PekingU/rtdetr_r18vd_coco_o365) has 300 queries,
256-dimensional tokens and three decoder layers.

## Integrated graph and supervision

Input remains original RGB `[B,3,384,576]` in `[0,1]`. Inside perception, a
differentiable bilinear resize produces 640x640, with no ImageNet normalization,
crop or padding. This follows the checkpoint's size/range policy; tensor
bilinear interpolation is explicit rather than silently invoking a PIL
processor. Normalized detector coordinates are converted from image y-down to
workspace y-up; physical workspace remains 0.6x0.4 m. Dataset pixels and poses
are not modified.

Native encoder top-300 proposal initialization is retained. It sets a fixed
model capacity, independent of GT count. No confidence filtering, NMS, GT top-k,
language selection mask, or token detach occurs between perception and layout.
All 300 final tokens and their predicted geometry enter the unchanged
three-stage Arrangement Decoder. Query row j always supplies both Source and
Target j. It is not a persistent tracking ID or a COCO category ID.

Source center uses the pretrained box center. Objectness logit is the maximum
of the full pretrained COCO class logits, with the existing sigmoid >=0.5
inference threshold. Classes are reused as foreground hypotheses, not mapped
to synthetic color/shape labels. Physical a/b, pi-periodic angle, color and
shape use the existing learnable heads, newly initialized for this backend.
These are not pretrained physical measurements. Generic bounding-box width/
height is never substituted for intrinsic object dimensions.

The same existing physical Detection loss is averaged over encoder proposals
and the three decoder outputs, using ONE final Source assignment for all four.
This supervises the pretrained proposal/classifier heads as well as the final
output. Target uses the same assignment and unchanged 0.25/0.25/0.5 stage
weights. No VFL/GIoU/denoising/collision loss is introduced. This is an adapted
project criterion, not a claim to reproduce the native RT-DETR training recipe.
With Q=300 the existing negative-weight normalization includes 300 queries;
its aggregate negative/positive balance differs from the legacy Q=12 model.

Native GT-conditioned denoising queries are not used in the joint forward.
Their pretrained embedding is retained but frozen and excluded from AdamW.
All other detector components remain trainable. Pretrained BatchNorm running
statistics stay fixed for S0 micro batches; affine parameters remain trainable.
Text encoder remains frozen and receives untruncated English tokens.

Native proposal-feature/reference and inter-layer reference stop-gradients are
removed in the adapter. Target gradients flow through tokens, box centers,
refinements, decoder, encoder and backbone. Discrete proposal-index selection
itself has no gradient; earlier score heads receive Detection auxiliary
supervision. No inference prediction is replaced by GT. At zero-initialized
arrangement residual heads, some layout paths only open after the first real
optimizer update; verification must inspect both initial and updated states.

## Deterministic sampling

Stock CUDA grid_sample backward fails with strict deterministic algorithms on
the pinned environment. The adapter locally replaces only deformable sampling
with four-neighbor, zero-padded bilinear sampling via gather. It has the same
coordinate convention as grid_sample(align_corners=False). Strict PyTorch
determinism remains enabled; no global monkeypatch or warn-only mode is used.
Regression tests compare values and gradients against the stock CPU operation,
and repeat CUDA backward exactly. Native detector forward equivalence is also
checked. This compatibility code is not a new detection algorithm.

## S0 preparation and stop boundary

Use `configs/s0_rtdetr.json` and `configs/pretrained_rtdetr.json`. The existing
seed 1203, FP32, effective batch 16 / micro batch 2, AdamW rates, clipping and
2,000-update maximum remain. The RT-DETR backbone receives backbone LR 1e-5;
pretrained encoder/decoder and new heads retain the configured other-module LR
1e-4. The architecture's 300 queries, internal checkpoint resize and auxiliary
Detection placement are explicit backend changes, not hidden hyperparameter
tuning. No old checkpoint is compatible with this new backend; start a fresh
run rather than weakening checkpoint provenance checks.

Lightweight preflight may make disposable optimizer updates and save a separate
test checkpoint. It must report zero-shot Source results honestly; COCO
pretraining does not guarantee coverage of synthetic Blender objects. A
technical GO for a controlled S0 experiment is not an S0 success claim.
Do not start a full S0 run until the compatibility/Source/gradient/memory report
has been presented. No S1 or new dataset generation is authorized.

Reproduce the CUDA preflight into a new, nonexistent directory:

```sh
conda activate knolling
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
python scripts/gpu_preflight_s0.py --config configs/s0_rtdetr.json \
  --report-dir artifacts/rtdetr/NEW_PREFLIGHT_DIRECTORY
RUN_PRETRAINED_TESTS=1 python -m pytest -q
```

The preflight performs three disposable optimizer calls: one 16-scene update,
one 9-scene tail update, then the same tail replay after checkpoint restoration.
It never invokes the training loop. It also saves native-forward parity,
all-32 zero-shot Source results, strict determinism evidence and Target-only
gradient probes. Peak memory includes Adam state and these diagnostic probes.
Use a fresh run directory and `configs/s0_rtdetr.json` when a full S0 run is
subsequently authorized; the preflight checkpoint is not its initialization.
