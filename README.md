# Knolling dataset pipeline

New implementation for `minjae35/knolling_bot_project`, based on the finalized
v1.2 specification in the project conversation. Scope: M0–M8 only. No neural
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

# Only run when generation is intended. Refuses a nonempty output directory.
conda run -n knolling python -m knolling_dataset build --count 100 --seed 1202 --output artifacts/pilot_v1_2

# Recheck hashes, identities, constraints, split leakage and all RGB/mask projections.
conda run -n knolling python -m knolling_dataset audit artifacts/pilot_v1_2
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
audit.json, COMPLETE.json            # COMPLETE only after full successful audit
scenes/<scene_family>/source/{rgb.png,instance_mask.png,render.json}
samples/<sample_id>/{annotation.json,target/...}
qc/{index.html,contact_sheet.png,summary.json,*_source.png,*_target.png}
work/{render_jobs.json,blender.log}
```

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

See [module and validation contracts](docs/dataset-contract.md) and
[implementation decisions](docs/implementation-decisions.md).
