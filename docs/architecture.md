# Architecture

v2.0.0 maps one overhead RGB image and one English sentence to a target pose for each detected object. Inference does not read a ground-truth pose, a teacher row, or a teacher order.

The image is 576 × 384. The model sees it as a float tensor `[B, 3, 384, 576]` with values in `[0, 1]`. Poses are metres and radians in a 0.60 m × 0.40 m workspace. At most eight objects are arranged.

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

A layout that already satisfies the gap, region, and row constraints is not rewritten. If no legal alternative exists, the original prediction is kept.

## Modules

The arrangement model is implemented in `src/knolling_model/arrangement/`. Perception checkpoints record the module names `knolling_model.recovery.perception`, `knolling_model.recovery.contracts`, and `knolling_model.recovery.data`. The files under `src/knolling_model/recovery/` re-export the corresponding modules from `arrangement`, so those checkpoints still resolve and their stored file hashes still match.

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

`experiments/end_to_end/score_end_to_end.py` defines `load_adopted`. That loader is the inference entry. The foreground threshold is 0.5.

`load_adopted` builds on `load_integrated`. That loader reads `configs/structural_integration.json` and calls `load_current`, which reads `configs/current_models.json`. The arrangement entry there points at `configs/recovery_v4_review.json` and the neural-keep checkpoint. The structural manifest then overlays the perception and arrangement bases. On top of that, `load_adopted` loads the relation head from `artifacts/phase1_relation_continue/checkpoint-best.pt` and the size head from `artifacts/experiments/joint_stage/checkpoint-best.pt`. The row module is initialized from the same joint checkpoint, then the selector placement and context weights in `artifacts/experiments/selector_full/checkpoint-best.pt` are loaded over it. Importing the scorer also reads `configs/phase1_pairwise_affine.json` and `configs/phase1_relation_pose.json`, because those files configure the relation runtime it imports. `configs/pretrained_rtdetr.json` selects the RT-DETR and MiniLM snapshots.

`experiments/end_to_end/demo.py` renders one full arrangement. It refuses partial and no-op. Those two modes go through `poses_from_detection`.

Weight paths and SHA-256 hashes are in [v2.0.0.md](v2.0.0.md).
