# Knolling dataset pipeline

New implementation for `minjae35/knolling_bot_project`, based on the finalized
v1.3 dataset specification, preserving the validated v1.2 pipeline. Dataset
modules M0–M8 are complete; S0 neural implementation is being added separately.
See the [vast.ai S0 setup and readiness runbook](docs/vast-ai-s0.md) and
[dataset storage audit and verified backups](docs/dataset-storage-audit.md).
Dataset transfer/storage changes remain pending approval; this repository does
not provision GPU instances.

**Initial system language: English only.** The active pilot is
`artifacts/pilot_v1_3_en/`. Both `pilot_v1_2` and the original bilingual
`pilot_v1_3` remain archived regression artifacts. Multilingual support is
optional future work; the Neural Architecture/text-encoder choice is unchanged.
See the [English conversion and validation report](docs/english-only-report.md).
The [final quality review](docs/final-dataset-quality-review.md) recommends retaining
the dataset for S0, subject to human sign-off. All-sample visual comparisons and
per-sample findings are in `artifacts/final_quality_review/index.html`; no labels
or Teacher behavior were changed by that review.

Use the existing Conda environment `knolling` (Python 3.12):

```sh
conda run -n knolling python -m pip install -r requirements-lock.txt
conda run -n knolling python -m pip install --no-deps --no-build-isolation -e .
conda run -n knolling python -m pytest -q
```

All geometry is in meters and radians. Pixel and normalized coordinates are
explicit derived values. Generated datasets and QC reports belong under the
Git-ignored `artifacts/` directory.

## Run and verify

Blender 5.1.0 must be on PATH, or set `BLENDER_EXECUTABLE` to its executable.
The host uses Python 3.12; Blender uses its own bundled Python. No BlenderProc,
GPU training framework, external asset download, or previous project code is required.

```sh
# Full tests: real repeated Blender renders AND a 10-scene builder integration.
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 conda run -n knolling python -m pytest -q

# One-time instruction-only conversion; refuses an existing output directory.
# Does not invoke Blender, regenerate scenes or recompute target arrangements.
conda run -n knolling python -m knolling_dataset english --parent artifacts/pilot_v1_3 --output artifacts/pilot_v1_3_en

# Recheck hashes, identities, constraints, split leakage and all RGB/mask projections.
conda run -n knolling python -m knolling_dataset audit --read-only
# Archival audits must opt out explicitly; this does not disable the gate on active English data.
conda run -n knolling python -m knolling_dataset audit artifacts/pilot_v1_2 --legacy --read-only
conda run -n knolling python -m knolling_dataset audit artifacts/pilot_v1_3 --legacy --read-only

# All regression/integration tests and all three on-disk pilots.
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 RUN_MIGRATION_TESTS=1 RUN_LANGUAGE_MIGRATION_TESTS=1 \
PILOT_DATASET=artifacts/pilot_v1_2 PILOT_V13_DATASET=artifacts/pilot_v1_3 \
PILOT_EN_DATASET=artifacts/pilot_v1_3_en \
conda run -n knolling python -m pytest -q
```

The CLI is capped at 100 scene families. A 100-scene pilot has 50 full, 30 partial,
20 no-op primary samples, plus 30 same-image counterfactual instructions: **130
samples, not 130 different scenes**. The 20 no-ops comprise 14 repeated requests
and 6 explicit keep-all requests. Keep-all preserves a safe scene even if its
objects are not axis aligned; repeated requests must satisfy their full semantics.

## Output contract

```text
asset_registry.json, generation.json, provenance.json
manifest.jsonl                       # one sample per line, annotation hash
train.jsonl, validation.jsonl, development.jsonl, test.jsonl
audit.json, AUTOMATED_VALIDATION.json # v1.3 automation only; human review pending
scenes/<scene_family>/source/{rgb.png,instance_mask.png,render.json}
samples/<sample_id>/{annotation.json,target/...}
qc/{index.html,contact_sheet.png,summary.json,*_source.png,*_target.png}
work/{render_jobs.json,blender.log}
```

Migration adds `parent_manifest.jsonl`, `parent_provenance.json`,
`parent_annotations/`, `parent_snapshot.json`, `source_file_hashes.json`,
`migration.jsonl`, `migration_summary.json`, `teacher_audit.jsonl`,
`counterfactual_audit.jsonl`, `partial_quality.jsonl`, `partial_quality_summary.json`,
`human_review.jsonl` and an empty `training_approved.jsonl`.
The legacy `COMPLETE.json` stays untouched in v1.2. v1.3 does not create that
marker or automatically authorize training. `generation.json` remains the
original generation history; migration counts are recorded separately.

For an explicitly intended fresh build in a **new** directory, `build` defaults
to `--teacher-policy balanced_rows_v1.3`. Use `--teacher-policy legacy_v1_2`
only for legacy reproduction. The Python builder/annotation APIs also default
to the balanced Teacher and English instructions. Unknown or
mixed policies are rejected. The structured Scene/Instruction schema stays 1.2;
policy and migration metadata are separate sidecars.

No-op targets reuse the exact source render; their target coordinates are still
fully serialized. Source and target use the same instance-ID mapping (1..N, zero
background), camera, lighting seed and physical assets. Masks are uint16 PNG;
ordinary image viewers may show them nearly black. RGB is sRGB uint8 PNG.
All descendants share source RGB and scene-family split. Scene-family split
counts are 70/10/10/10; asset-family pools are strictly disjoint. This is a small,
strict geometry-OOD pilot, not evidence of future neural generalization.

Open `qc/index.html` locally to filter by mode/split and inspect input/target
pairs, IDs, locks and English instructions. The JSON report is automatic
verification; **human visual/language review remains a separate sign-off**.

The active report covers **all 130 samples**, with raw/annotated RGB switching,
source/target views, movement arrows and per-object translation/rotation.
Use `qc/english_instructions.csv` and `qc/object_movements.csv` for review.
Historical geometry comparisons remain in the archived v1.3 QC report.
The contact sheet is only a navigation aid. Human ratings remain null until a
reviewer records them; this phase does not implement a training export workflow.

The English conversion changes only `instruction.text` and `instruction.language`.
All source/target geometry, identity, family and split fields remain identical;
RGB/mask/render-metadata files retain their hashes. Current language statistics
are in `language_migration_summary.json`; inherited `generation.json` and geometry
migration reports describe historical runs, not the active language distribution.
`language_parent_*` and `parent_annotations/` are archival evidence only. Future
loaders must read the active manifest/split indexes, never glob every annotation.

The audit requires `language="en"` plus exact agreement with the approved English
grammar and structured fields. A non-English string relabeled `en`, foreign text
using ASCII, or a contradictory gap/direction is rejected. English v1 is a bounded
instruction interface, not a claim of unrestricted English understanding. For example:

> Keep every object at its current position and orientation.

For arrangements, the template names the selected objects and region, long-axis
direction, bounding-rectangle area order, row/color grouping and exact horizontal
and vertical row-band gaps. Partial instructions explicitly keep other objects fixed.

See [module and validation contracts](docs/dataset-contract.md) and
[implementation decisions](docs/implementation-decisions.md).
The v1.3 policy and migration choices are in
[v1.3 implementation decisions](docs/v1.3-implementation.md).
The completed migration counts, verification and remaining quality gaps are in
[the v1.3 pilot report](docs/v1.3-pilot-report.md).

The completed 100-scene pilot's results, milestone commits and limitations are
documented in [the pilot report](docs/pilot-report.md).
