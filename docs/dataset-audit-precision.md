# Movement-score audit precision investigation

Date: 2026-11-03. Starting code: `cb14f74`, clean working tree. The audit was
run against the unchanged English pilot at `artifacts/pilot_v1_3_en`, using
Python 3.12.15 on Linux x86_64/glibc 2.39 and the isolated pinned neural CPU
environment `/tmp/knolling-s0-review-venv`. Dataset provenance records Python
3.12.16 and the original Mac repository paths.

## Reproduction and numerical evidence

The complete read-only audit reproduced exactly these five failures:

1. `scene_0037_primary:teacher_scores_or_configuration`
2. `scene_0089_primary:teacher_scores_or_configuration`
3. `scene_0037_primary:migration_scores`
4. `scene_0089_primary:migration_scores`
5. `partial_quality_mismatch`

Only two numeric values differ after JSON normalization of the recomputed
sidecars. The fifth failure is entry 11, `scene_0037_primary`, field
`v1_3.J_move` in `partial_quality.jsonl`; the same value appears in the Teacher
and migration sidecars. All other fields match. Old migration scores match.

| Sample | Stored J_move (m²) | Recomputed Teacher and layout audit (m²) | Computed − stored (m²) | Binary64 steps |
|---|---:|---:|---:|---:|
| `scene_0037_primary` | 4.440863602014575e-06 | 4.440863602014576e-06 | 8.470329472543003e-22 | 1 |
| `scene_0089_primary` | 0.05376727341039903 | 0.05376727341039904 | 6.938893903907228e-18 | 1 |

The first sample is Train/Partial, four objects with one selected (`o02`), and
is in the S0 selection. The second is Train/Full with seven selected objects
and is outside the N=2–4 S0 selection. Relative differences are respectively
1.9073608720386012e-16 and 1.2905422692617309e-16. Their hexadecimal values are
respectively `0x1.2a0574abcf1b4p-18` versus `0x1.2a0574abcf1b5p-18`, and
`0x1.b876251c91715p-5` versus `0x1.b876251c91716p-5`.

## Cause and independent calculation

The Teacher computes the selected-object mean squared translation:
`fsum(dx**2 + dy**2 for selected objects) / selected_count`. The layout audit's
`layout_metrics` delegates to that same `movement_cost`; there is no separate,
inconsistent production formula. `teacher.py`, `teacher_balanced.py`, and
`quality.py` are byte-identical to the dataset's archived Teacher revision
`400d9c487fc886dd8ec9ca13f6ae7379ad20fc1e`. Replaying that unchanged computation
in this Linux runtime gives the recomputed column above.

Independent calculations from the source/target coordinates isolate the scale
and mechanism of the discrepancy:

- For sample 0037, `dy=-0.001911491438566365`. On this runtime, `dy**2` gives
  `3.653799519712512e-06`, while `dy*dy` gives
  `3.6537995197125115e-06`. Using multiplication for the squares reproduces
  the stored final cost exactly.
- For sample 0089, summing all 14 squared components in one `math.fsum`, rather
  than rounding each x²+y² pair before summation, reproduces the stored final
  cost exactly. The production Teacher still uses its original grouping.
- A separate 100-digit Decimal calculation from the exact binary64 coordinate
  inputs, with high-precision subtraction/squaring/summation, rounds to
  `4.440863602014576e-06` and `0.05376727341039903` respectively. Thus neither
  replacing every stored score nor declaring one platform's arithmetic the
  unique Ground Truth would be justified.
- JSON encode/decode preserves each stored and computed float exactly. The
  discrepancy is in recomputation, not a lossy decimal serialization step.

These are last-bit floating-point reproducibility differences in a derived
score. The original Mac intermediate arithmetic was not archived, so the
specific historical library/compiler operation responsible cannot be replayed
here. The unchanged production formula, exact input/target reproduction, and
independent rounding experiments establish numerical precision as the cause;
there is no evidence of dataset corruption or a changed Teacher algorithm.

## Verifier correction

Only the v1.3 audit changes. At its three comparison sites—Teacher audit,
migration metrics, and the `legacy`/`v1_3` metrics in Partial quality reports—a
positive `J_move` may differ by exactly one adjacent binary64 value, tested
with `math.nextafter`. This is the smallest nonzero representable allowance;
it introduces no meter-based absolute threshold or broad relative tolerance.
Two-step changes fail. Zero (including its existing representation), negative,
nonfinite, and non-float score changes fail. All other fields and keys remain
exact, including `J_shape`, object movement diagnostics, IDs, policy/config,
counts, template selection, and hashes.

No score is rounded or written back to the dataset. No Target, Teacher,
geometry threshold, Teacher ranking key, model, loss, sampler or training
configuration changes. The Teacher's existing 12-decimal candidate ranking is
unchanged; both affected scores already have identical rounded ranking keys.

## Integrity and S0 impact

Before the correction, all other complete-audit checks passed: 130 samples,
100 source families, expected annotation/render/source/archive hashes, identity
and split checks, geometry, English semantics, all 30 counterfactual pairs, and
130 successes in each of policy reproduction, array-order invariance, language
invariance, asset-alias invariance, and No-op reapplication. A separate replay
compared raw ID-indexed target poses, without the audit's rounded target hash:
all 130 targets are exactly equal, with zero identity or raw-pose differences.

`J_move` is a dataset quality/Teacher diagnostic. It does not enter S0 RGB/text
inputs, supervision tensors, Detection + Target loss, checkpoint selection,
evaluation success criteria, or the memorization gate. The stored target poses
and object correspondence remain unchanged, so these score differences cannot
change S0 training or evaluation. The failure was the verifier's exact
comparison of recomputed diagnostics.

## Validation and recommendation

- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /tmp/knolling-s0-review-venv/bin/python scripts/verify_s0_data.py`:
  **valid**, 130 samples / 100 families / 32 S0 samples. This invokes the complete
  audit with render verification and policy invariance checks before the S0
  adapter. Manifest SHA-256 remains
  `8000428a04522f50cf37567d85d442ac94d1f8a87121747d4243e33974320d06`.
- `PILOT_EN_DATASET=artifacts/pilot_v1_3_en PYTHONDONTWRITEBYTECODE=1 /tmp/knolling-s0-review-venv/bin/python -m pytest -q -rs -p no:cacheprovider tests`:
  **132 passed, 8 skipped**, including the delivered English pilot acceptance
  test and 15 new precision/tampering regression cases. Skips are pretrained
  cache and opt-in rendering/migration/absent legacy-pilot integrations.
- New tests accept the two observed one-step differences in either direction;
  reject two steps, changes at zero, invalid types/nonfinite/negative scores,
  unrelated metric/coordinate/identity/config/hash changes; exercise the actual
  Teacher audit with synthetic sidecars; and check no comparison mutates input.
- SHA-256 snapshots of **all 1,750 dataset files** before and after the full
  checks are identical, with no added, removed or changed files. The aggregate
  sorted path/hash JSON SHA-256 is
  `b1f1c332db11966f64cd956e03bf071647174fa7b9121a2961af846c451711bc`.
- Git diff and whitespace checks pass. Changes are confined to the verifier,
  regression tests and documentation.

**GO for GPU Preflight:** the five dataset audit blockers are resolved without
changing data or Teacher behavior. Actual CUDA compatibility, pretrained-cache
readiness, memory requirements and convergence still require their respective
checks. This recommendation is not a training result; no GPU Preflight or S0
training command was run during this investigation.
