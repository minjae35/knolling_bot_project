# Architecture

v2.0.0 maps one overhead RGB image and one English sentence to a target pose
for each detected object. Inference does not read a ground-truth pose, a
teacher row, or a teacher order.

The image is 576×384. The model sees it as a float tensor `[B, 3, 384, 576]`
with values in `[0, 1]`. Poses are metres and radians in a 0.60 m × 0.40 m
workspace. At most eight objects are arranged.

## Data flow

```text
BEV RGB ──► RT-DETR-R18
              center, size, angle, color, shape
              duplicate suppression at 10 mm

English ──► MiniLM text encoder ──► 13-d goal
              region and gap are pinned to the sentence
              only when the goal's argmax disagrees

full ──► relation features
      ──► row and order selector
      ──► constrained decode, at most 8 candidates
      ──► area order
      ──► geometry decoder
      ──► row repair, only if the layout is infeasible
      ──► single-color split, only if the sentence asks for it,
          colors are confident, and a predicted row mixes colors

partial / no-op ──► move the named color, or keep every object
                    held objects stay on the detected pose
```

A layout that already satisfies the gap, region, and row constraints is not
rewritten. If no legal alternative exists, the original prediction is kept.

## Modules

| Piece | Code | Role |
| --- | --- | --- |
| Detector | `src/knolling_model/rtdetr.py`, `perception/` | RT-DETR-R18, 300 queries |
| Text and goal | arrangement model loaded by `load_integrated` | sentence to a 13-d command |
| Integration | `src/knolling_model/experiments/integration.py` | perception and arrangement base |
| Relation head | `src/knolling_model/experiments/relation_pose.py` | features for the selector |
| Selector | `src/knolling_model/experiments/set_context.py` | row and slot scores |
| Constraints | `src/knolling_model/experiments/constraint_decode.py` | legal rows, repair, color split |
| Geometry | `src/knolling_model/experiments/geometry_decoder.py` | `x`, `y`, axis |
| Partial and no-op | `src/knolling_model/experiments/partial_arrangement.py` | named color, or keep all |
| Entry point | `poses_from_detection` in `size_refinement.py` | the three modes |

`experiments/end_to_end/score_end_to_end.py` defines `load_adopted`. That
loader is the inference entry. The foreground threshold is 0.5.

`experiments/end_to_end/demo.py` renders one full arrangement. It refuses
partial and no-op. Those two modes go through `poses_from_detection`.

## What is still in the tree, and why

`src/`, `scripts/`, and `configs/` also contain the recovery and phase-1
training stack. v2 still imports parts of it while loading weights:

- `configs/structural_integration.json` names the perception and arrangement bases.
- `configs/current_models.json` and `configs/recovery_v4_review.json` are read
  before those bases are overlaid.
- `configs/pretrained_rtdetr.json` selects the RT-DETR and MiniLM snapshots.
- Importing the scorer also reads `configs/phase1_relation_pose.json` and
  `configs/phase1_pairwise_affine.json`.

Those files stay where they are so the checkpoints still load. Moving them
would change the import path. Historical write-ups are under `archive/docs/`.

Weight paths and SHA-256 hashes are in [v2.0.0.md](v2.0.0.md).
