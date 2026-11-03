# RT-DETR integration verification — 2026-11-03

Technical **GO for a controlled 32-sample S0 experiment**. No full S0 training
was started. Zero-shot detection and the new physical/layout heads do not yet
meet the S0 gate; this is not a convergence or generalization claim.

The implementation contract, checkpoint revision/hash, graph changes and exact
configuration are in [rtdetr-integration.md](rtdetr-integration.md).

Regression validation: `RUN_PRETRAINED_TESTS=1 python -m pytest -q -rs`
passed **143 tests**, with **8 opt-in checks skipped** (rendering, archived
dataset/migration acceptance and environment-gated English acceptance). Real
cached pretrained forward/backward tests ran for both perception backends,
with network access forbidden. The separate complete English dataset audit
ran successfully in CUDA preflight. No Blender render or migration was run.

## Evidence

Local machine-readable report:
`artifacts/rtdetr/preflight-20261103T122000Z/report.json`.
All-32 query scores, native boxes, diagnostic Source assignments and errors:
`artifacts/rtdetr/preflight-20261103T122000Z/source-detection.json`.
These generated artifacts and the disposable checkpoint are ignored by Git.

| Check | Result |
|---|---|
| GPU/runtime | One NVIDIA GeForce RTX 4090, capability 8.9; torch 2.9.1+cu128; torchvision 0.24.1+cu128 |
| Environment | Existing Python 3.12 `/venv/knolling`; no dependency or driver changes; pip check passes |
| Assets | Detector and MiniLM pinned revisions and LFS SHA-256 verified; no missing, unexpected or mismatched detector keys |
| Dataset | Complete audit passes: 130 samples, 100 families, all 32 S0 samples |
| Stock backward | Confirmed strict-determinism grid_sample failure |
| Adapted backward | Strict determinism retained; finite full-resolution forward/backward |
| Native forward parity | Maximum normalized-box difference 2.563e-6, logit difference 1.154e-4, token difference 5.329e-5 |
| Target-only backward | Nonzero finite gradients in backbone, encoder, proposal features/boxes, decoder, physical heads, arrangement and text projection |
| Unfiltered tokens | All 300 token rows receive Target gradients; frozen MiniLM has none |
| Repeatability | Target-only CUDA gradients bitwise identical across repeated probes |
| Optimizer | Three disposable AdamW calls with configured accumulation: 16 scenes, 9 scenes, restored 9-scene replay |
| Resume | Model/optimizer load and next-update loss/model/optimizer replay bitwise exact |
| Peak allocated | 2,334,063,104 bytes, approximately 2.174 GiB |
| Peak reserved | 2,952,790,016 bytes, 2.750 GiB |
| Adam state | 191,639,356 bytes; peaks include optimizer state and diagnostics |
| Preservation | All 1,750 dataset files unchanged; 1,802 dataset/previous-run/report files hash-verified unchanged |

Sampling regression tests compare stock CPU forward and gradients with respect
to values, locations and attention weights, including borders and repeated
sample locations. CUDA collision/reduction backward repeats exactly. Small
real RT-DETR unit models additionally check coordinates, same-query identity,
normalization, optimizer coverage and single-assignment auxiliary supervision.

## Source test — no fine-tuning

At the existing 0.5 threshold, the pretrained detector emits 50 objects for 93
GT instances across the 32 instruction samples (25 unique Source families).
Seven scenes have correct counts: **21.875%**.

Native axis-aligned box diagnostics give 45 one-to-one IoU>=0.5 matches:
precision **90.0%**, recall **48.387%**. These are not the evaluator's rotated
physical-pose identity gates. Under the GT-assisted project Source assignment,
center P95 is **1.271 mm**. Matched/unmatched score medians are **0.4994/0.00187**.
The center diagnostic includes low-confidence queries and cannot replace count
or identity admission. Native COCO scores still under-detect synthetic shapes.
No threshold was tuned to these samples.

Physical size/orientation/color/shape heads and Arrangement Decoder weights
are initialized anew. They are not asserted to be pretrained or correct.
The first effective-batch pre-update loss was 13.9528; it is not a final all-32
learning curve or proof of overfitting.

## Remaining risks and next boundary

1. Synthetic-object domain adaptation and foreground confidence still need the
   controlled S0 experiment. Keep the existing count/pose success criteria.
2. Physical orientation/size and language-conditioned Target learning remain
   unproven; successful pretraining transfer alone cannot guarantee them.
3. This backend preserves pretrained detection weights but uses the project's
   physical Detection + Target criterion, not native VFL/GIoU/GT denoising.
   The fixed auxiliary mean and Q=300 negative normalization are documented
   differences; no loss-weight search was performed.
4. The adapter's continuous refinement gradients and bilinear sampling are
   tested on the pinned stack. Future Transformers updates need a new review.
5. Measured memory uses FP32, micro batch 2 / effective batch 16. It does not
   establish memory use for larger images, batches or S1.

Next: a fresh S0 run using `configs/s0_rtdetr.json`, with existing seed/budget,
separate persistent logs/checkpoints and unchanged evaluation. Stop at that
experiment; no larger training or new dataset generation.
