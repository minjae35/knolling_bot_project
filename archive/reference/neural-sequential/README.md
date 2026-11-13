# neural_sequential_xy_v1 reference snapshot

This directory freezes `neural_sequential_xy_v1` from commit
`4019217954eaa8b9d9a322ad2bc48d278e0bca2c` on
`experiment/neural-sequential-phase1`.

It is not the current end-to-end architecture. Do not connect it to
`current.py`, the default model, or any other live entry point. Inference
must not read ground truth, a rule planner, or a geometry solver.

Rejected sibling experiments were left out on purpose. This snapshot does not
include direct-centre, relational row/slot, StructDiffusion, the single-row
decoder, the gap auxiliary loss, the teacher-prefix X-offset loss, the
size-conditioned offset head, the Set Network, or pairwise XY.

## Files included

- `neural_sequential.py`: identical to `src/knolling_model/experiments/neural_sequential.py` at that commit. `ScenePoseDecoder` lives in this file.
- `test_neural_sequential.py`: a copy of that commit's unit test. It is not part of the test suite that is run. It imports `Reference`, `example`, and `device` from `test_phase1_single_row`, and it imports the live path `knolling_model.experiments.neural_sequential`. Those dependencies were not copied.
- `docs/neural-sequential-phase1.md`
- `configs/neural_sequential_phase1.json`

## Files not included

- `scripts/run_neural_sequential.py` was not copied because it imports `knolling_model.experiments.direct_center`, `knolling_model.experiments.phase1_evaluation`, `run_phase1_single_row_generalization`, `run_phase1_single_row_overfit`, and `verify_phase1_single_row`.
- `tests/neural/test_neural_sequential_driver.py` was not copied because it imports that runner.
- `offset_supervision.py`, `size_conditioned_offset.py`, `horizontal_gap.py`, and untracked experiment runners were not committed because those experiments were rejected.

## Imports this module expects

`neural_sequential.py` imports failed experiment modules directly. Those files
were not put back under `src/`.

- `.generalization.new_model`
- `.phase1_single_row.position_loss` (`position_loss` is also re-exported under this module's name)

The only other imports that were already in use then are
`knolling_dataset.builder.file_hash` and
`knolling_model.recovery.storage.load_artifact`.
