# Knolling dataset pipeline

New implementation for `minjae35/knolling_bot_project`, based on the finalized
v1.3 dataset specification, preserving the validated v1.2 pipeline. Scope: M0–M8 only. No neural
network, training pipeline, or robot simulation.

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

# Migrate the original fixed inputs; do not regenerate the original pilot.
conda run -n knolling python -m knolling_dataset migrate --parent artifacts/pilot_v1_2 --output artifacts/pilot_v1_3

# Recheck hashes, identities, constraints, split leakage and all RGB/mask projections.
conda run -n knolling python -m knolling_dataset audit artifacts/pilot_v1_2 --read-only
conda run -n knolling python -m knolling_dataset audit artifacts/pilot_v1_3 --read-only

# All regression/integration tests and both actual on-disk pilots.
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 RUN_MIGRATION_TESTS=1 \
PILOT_DATASET=artifacts/pilot_v1_2 PILOT_V13_DATASET=artifacts/pilot_v1_3 \
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
only for legacy reproduction. The Python API retains its legacy default for
backward compatibility; specify the policy for every new caller. Unknown or
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
pairs, IDs, locks and Korean/English instructions. The JSON report is automatic
verification; **human visual/language review remains a separate sign-off**.

The migrated report covers **all 130 samples**, with raw/annotated RGB switching,
source/v1.2/v1.3 targets, movement arrows and per-object translation/rotation.
Use `qc/target_comparison.csv`, `qc/object_movements.csv`,
`qc/partial_quality.csv` and `qc/partial_quality.md` for detailed comparisons.
The contact sheet is only a navigation aid. Human ratings remain null until a
reviewer records them; this phase does not implement a training export workflow.

See [module and validation contracts](docs/dataset-contract.md) and
[implementation decisions](docs/implementation-decisions.md).
The v1.3 policy and migration choices are in
[v1.3 implementation decisions](docs/v1.3-implementation.md).
The completed migration counts, verification and remaining quality gaps are in
[the v1.3 pilot report](docs/v1.3-pilot-report.md).

The completed 100-scene pilot's results, milestone commits and limitations are
documented in [the pilot report](docs/pilot-report.md).
