# English-only v1 dataset completion report

The active dataset is `artifacts/pilot_v1_3_en/`, derived from the original
`pilot_v1_3` without changing task semantics or target arrangements. English is
the only supported initial-system language. Multilingual support is optional
future work; the Neural Architecture and current text encoder are unchanged.

## Delivered data

| Check | Result |
|---|---:|
| Active English samples | 130 / 130 |
| Original languages | 73 Korean, 57 English |
| Scene families | 100 |
| Train / validation / development / test | 89 / 13 / 13 / 15 |
| Texts changed | 127; three Keep-all texts already matched |
| Structured semantic inconsistencies | 0 |
| Source or target poses changed | 0 |
| Source or target views re-rendered | 0 |
| Unchanged active RGB/mask/render-metadata files | 630 |
| English instruction length | 9–89 words |
| Human review | Pending for all 130 samples |

Only `instruction.text` and `instruction.language` differ in active annotations.
All other annotation fields are preserved, including identities, assets, selected
and locked sets, geometry, relations, scene families and splits. English wording
is produced deterministically from structured fields and observable source
references. It retains area ordering, axis direction, exact horizontal/vertical
gaps, color-row purity/contiguity, region and lock semantics. No-op arrangements
retain their original request rather than being relabeled as Keep-all.

The English audit rejects foreign text even if its language field is `en`, and
rejects wording inconsistent with the structured fields. This is an exact bounded
grammar check, not a general-purpose language detector or evidence of a neural
model's language understanding. The Teacher text-invariance check likewise does
not demonstrate learned paraphrase understanding.

## Preservation and provenance

The complete original trees remain byte-for-byte unchanged against pre-migration
snapshots: 1,041 files in `pilot_v1_2` and 1,614 files in `pilot_v1_3`. The new
artifact uses independent copies, not hard links to mutable legacy files.

Manifest SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Original v1.2 | `b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27` |
| Original v1.3 | `515292bcff7d5ed7bcf8cd5edf4a3e13ea4485e581d9c378aaac6dc9b7dd7d9c` |
| Active English v1.3 | `8000428a04522f50cf37567d85d442ac94d1f8a87121747d4243e33974320d06` |

The English artifact records `language_policy="english_only_v1"` and
`instruction_contract="english_rows_v1"`; the Teacher remains
`balanced_rows_v1.3`. Migration code revision is `5025e70`. Annotation hashes,
the manifest and all four split indexes were updated. Language changes and hashes
are recorded in `language_migration.jsonl` and `language_migration_summary.json`.
Original generation and geometry-migration reports remain historical records.

Archived parent annotations retain their original languages for lineage checks.
They are explicitly excluded from active data. Future loaders must use the active
manifest/split indexes, never recursively collect annotation files. The human
review records remain pending and `training_approved.jsonl` is empty.

## Validation and review

Milestone `06c2cf6` passed 70 tests with four opt-in integrations skipped; focused
English/legacy instruction checks passed 30 tests after the wording refinement.
Milestone `5025e70` passed 73 tests with five opt-in checks skipped. Its real
130-sample migration integration forbids renderer and scene-generator calls,
checks exact semantic preservation and independent image copies, and confirms
that a tampered non-English sample is rejected even after hashes are repaired.

The final complete suite passed **78 tests, zero failures and zero skips in
170.48 seconds** on 2026-11-03. This includes every existing regression, real
Blender reproducibility, both builder policies, geometry migration, text-only
migration and acceptance audits of all three actual datasets:

```sh
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 RUN_MIGRATION_TESTS=1 RUN_LANGUAGE_MIGRATION_TESTS=1 \
PILOT_DATASET=artifacts/pilot_v1_2 PILOT_V13_DATASET=artifacts/pilot_v1_3 \
PILOT_EN_DATASET=artifacts/pilot_v1_3_en \
/Users/minjae/miniconda3/envs/knolling/bin/python -m pytest -q
```

Temporary rendered fixtures belong to the integration tests; the delivered
English migration itself renders zero views. The default CLI read-only audit
also passes against the English artifact. Git whitespace/diff checks pass.

The active audit passes geometry, collisions, boundaries, identity, instructions,
render projections, split integrity and language lineage. All 130 policy
reproductions, array-order checks, text-invariance checks, asset-alias checks and
same-selection No-op reapplications pass. All 30 existing counterfactual pairs
remain diagnostic under the independent structured predicates.

Review `artifacts/pilot_v1_3_en/qc/index.html` for all 130 English instructions,
source/target images, raw-image toggles, movement overlays and per-object movement
tables. CSV exports are `qc/english_instructions.csv` and
`qc/object_movements.csv`; machine checks are in `audit.json`.

Automated preservation checks found no semantic inconsistencies. Human visual
and language-quality approval is still required. Existing Partial coverage gaps
are unchanged: this migration does not add multi-object/organized-source cases
or silently change the distribution. No Neural Network training or full-scale
dataset generation was performed. Generated artifacts remain local and excluded
from Git.
