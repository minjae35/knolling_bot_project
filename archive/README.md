# Archive

These files are kept so earlier experiments can still be read. They are not the
v2.0.0 system.

- `docs/` holds the S0, recovery, perception, StructDiffusion, and v1.3 notes
  that used to sit in `docs/`. The same notes are also on
  `archive/e2e-failed-experiments`. Korean text is the original record.
- `experiments/` holds rejected recovery helpers that are not on the inference path.
- `reference/neural-sequential/` is a frozen copy of an earlier pose decoder.
  It is not the live package. The tests that still name
  `knolling_model.experiments.neural_sequential` fail to collect because that
  module is not in `src/`.

`archive/docs/git-identity.md` is the local Git identity note. It is not part
of the model. The agent checklist `AGENTS.md` is kept on disk and gitignored.

Do not treat numbers in `docs/` as the v2 holdout result. That result is in
`docs/evaluation.md`.
