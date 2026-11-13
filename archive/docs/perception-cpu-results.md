# Independent Perception CPU results — 2026-11-08

Implementation and CPU verification are complete in the separate
`codex/perception` worktree. The user explicitly selected CPU-only work because
there is no allocated exclusive GPU window. No GPU training was started.

## Evidence and implementation

- [Interface, data contract, commands and future training boundary](perception-independent.md)
- [Full annotation audit](perception-evidence/annotation-audit.json)
- [Actual-model CPU report](perception-evidence/cpu-report.json)
- [Initial Validation metrics](perception-evidence/validation-initial.json)
- [RGB-only CLI output](perception-evidence/rgb-inference.json)

Pilot's 130 annotation records contain 100 distinct source families. Recovery's
76 records reuse 25 existing Train families and contribute no new RGB sources.
Every source RGB, instance mask, mask BBox, calibrated pose, dimension and
physical-envelope annotation passed hash/integrity and independent projection
checks. Recovery's missing duplicated detection fields are resolved by exact
source/RGB joins to existing Pilot annotations, without generating annotations.

Train uses 70 source families / 341 objects; Validation uses 10 / 56. The
existing family assignments are unchanged, with zero family overlap. Reserved
Test and Development sources were annotation-audited but not prediction-evaluated.

## Measured baseline

The pinned COCO/Objects365 RT-DETR geometry is loaded, but task-specific
foreground and physical heads are **newly initialized and untrained**. These
numbers describe that initialization, before the disposable CPU verification
update; they are not trained performance or native COCO-model results.

| Validation measurement | Result |
| --- | ---: |
| Sources / annotated objects | 10 / 56 |
| Count Accuracy at confidence 0.5 | 0 / 10 = 0% |
| Count MAE | 5.6 objects |
| TP / FP / FN at confidence 0.5, envelope IoU >=0.5 | 0 / 0 / 56 |
| Precision / Recall | 0 / 0 |
| AP50 (0–1) | 0.0021320132 |
| AP75 (0–1) | 0.0005948525 |
| mAP50:95 (0–1) | 0.0008804309 |
| Operational position, major/minor dimension, angle errors | Unavailable: zero matched detections |

AP considers all scores on 300 queries/image with the documented class-agnostic
101-recall-point protocol. Very low nonzero AP alongside zero detections at 0.5
is expected: AP evaluates the full score ranking, not just predictions above
the operating threshold. No threshold was tuned to produce nonempty outputs.
The separate GT-assisted assignment diagnostic in the JSON is explicitly
excluded from operational accuracy and checkpoint selection.

## CPU verification

The real full RT-DETR-R18 preflight loaded all 100 sources and used actual
2-object and 8-object Train scenes for Forward/Backward. Native detection,
denoising and physical losses were finite; total loss was 33.9765548706.
Backbone, encoder, proposal score/box, decoder, foreground, physical and
denoising gradients were finite and nonzero. One disposable AdamW update
verified the optimizer/checkpoint path. Strict restore produced exactly
identical source tensors (maximum difference 0). The final preflight took
28.47 seconds on CPU.

All **30 distinct tests passed**. The selection initially passed 28 and skipped
two existing data tests because the worktree lacked their relative Pilot
fixture path. Both subsequently passed after exposing the same immutable Pilot
through an ignored local link. The changed checkpoint compatibility test also
passed independently. Tests cover corrupted/missing annotations, family
leakage, units/angles, empty/capacity cases, Oracle provenance rejection,
RGB-only inference, detection metric edge cases, optimizer/RNG reproduction,
adapter-independent weight reuse, GPU allocation guards, and fixed update
limits/best retention using a CPU test double.

The CLI independently reloaded `initial.pt`, performed RGB-only inference, and
re-evaluated the full Validation set. Every metric exactly matched the initial
report. This confirms executable inference/evaluation rather than just test
fixtures. Real confident detections remain empty until training succeeds.

## Checkpoints

Local artifacts are in
`/root/.codex/worktrees/perception/knolling_bot_project/artifacts/perception/cpu-final/`.
Small evidence JSONs are committed; binary checkpoints are ignored by Git and
remain local to this host/worktree.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `initial.pt` | Untrained baseline; zero optimizer updates | `c9c1c3a19f4ce2a3a8a59f0726fbf2d5a28d831ee7310372e1ec4625a40fe820` |
| `cpu-preflight.pt` | Disposable CPU gradient/reload verification; one update | `07f2418b77279a35e1baa2a70d9c6a91be5c9e755d0c6e42d0d224b988a7e15e` |

There is **no trained Best Checkpoint**. A future authorized single training run
will preserve `best.pt` and `last.pt` separately, including the initial best
candidate. It will start from pinned pretrained weights and fresh task heads,
not the disposable preflight update.

## Remaining blockers

1. An allocated exclusive GPU slot and one fixed-budget training run. The
   shipped configuration rejects training; the future proposal is capped at
   500 updates / 1,800 seconds, without search or retries.
2. Trained detection/pose accuracy and held-out generalization. Seventy unique
   Train views are small; new annotations were not invented to enlarge them.
3. Arrangement architecture stability before final end-to-end/joint training.
   The versioned independent object JSON and separate current-Objects adapter
   are ready; no end-to-end accuracy claim is made.
4. Real-camera calibration/domain evaluation before applying this synthetic BEV
   model outside the verified dataset.

Arrangement code, branches, checkpoints, configuration and active processes
were not changed. No notification/progress worker or direct Telegram transport
was introduced. Commits use the checkout's configured Author and Committer:
Minjae Bang <mb4882@columbia.edu>.
