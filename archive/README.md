# Archive

These files are kept so earlier experiments can still be read. They are not the
v2.0.0 system.

- `experiments/` holds rejected recovery helpers that are not on the inference path.
- `reference/neural-sequential/` is a frozen copy of an earlier pose decoder.
  It is not the live package. The tests that still name
  `knolling_model.experiments.neural_sequential` fail to collect because that
  module is not in `src/`.

The S0, recovery, perception, and v1.3 notes are not in this branch. They remain
on `archive/e2e-failed-experiments`. Do not treat numbers there as the v2
holdout result. That result is in `docs/evaluation.md`.
