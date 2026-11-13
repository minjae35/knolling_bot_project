# Dataset contracts and implementation order

The geometry rules below still describe the workspace. The closing paragraphs
about a 100-family cap, and the sentence that no training has been run, are
the original pilot. The 20,000-scene set and the v2 holdout are in
[dataset.md](dataset.md).

Active v1 language policy: **English only**, with `language_policy="english_only_v1"`
and `instruction_contract="english_rows_v1"`. `pilot_v1_3_en` is the active artifact;
original v1.2/v1.3 datasets and bilingual fixtures are archived regression evidence.
The schema still reads legacy `ko` values, but the active audit rejects them.

| Module | Entry point | Input → output | Dependencies |
|---|---|---|---|
| M0 | `schema.py` | Immutable assets, poses, scenes, instructions; JSON conversions | Standard library |
| M5 | `validate_source`, `validate_layout` | Source + target + instruction → errors, geometry metrics | M0, independent Shapely geometry |
| M1 | `build_registry`, `make_asset` | Seed, integer mm dimensions → asset pools / footprints | M0 |
| M2 | `generate_scene` | Pool, seed, split, count, mode → source Scene | M0, conservative SAT sampling |
| M3 | `generate_instruction` | Scene + seed + mode/options → text and semantic constraints | M0 |
| M4 | `solve` | Scene + Instruction → status, target, row/relation labels | M0, placement geometry, independent M5 final check |
| M6 | `render_batch`, `validate_render` | JSON jobs → RGB, masks, metadata; independent projection report | Blender bpy; host Pillow/NumPy/Shapely |
| M7 | `prepare_annotations`, `build_dataset` | Count/seed/output → accepted annotations, manifest, renders, QC | M1–M6 |
| M8 | `audit_records`, `audit_dataset` | Records or output root → leakage/integrity report, split indexes | M0, M5, M6; never trusts stored success flags |

M0 and M5 were implemented before the Teacher. All geometry is in **meters** and
angles in **radians**. There is no model loss or predicted size in this phase.
Annotation centers `uv` are dimensionless, `px` are pixel centers (top-left pixel
center 0,0), and `xyxy_px` mask boxes have exclusive maximum bounds.

Workspace: 0.600 × 0.400 m. Origin: bottom left. +x right, +y up, angle CCW from
+x, major-axis direction modulo pi. Image: 576 × 384. Conversion:
`u=x/0.6`, `v=y/0.4`, `px=576*u-0.5`, `py=384*(1-v)-0.5`.
`source_envelope_xyxy_uv` uses world-frame bottom-left uv bounds, not image y.
Envelope = oriented major/minor bounding rectangle. Mesh footprint is stored
separately and used for true mask projection. All collision and boundary labels
use the conservative envelope, not the smaller chamfer/capsule footprint.

## Geometry and semantics

- 2–8 fully visible objects; box/chamfer/capsule, aspect ratio >= 1.3.
- a=40–80 mm, b=25–50 mm, h=5–20 mm, dimensions on a 1 mm grid.
- Source: >=5 mm pair clearance, >=10 mm boundary margin. Partial and explicit
  keep-all generation deliberately use a stricter 12 mm clearance/margin.
- Every accepted target: >=10 mm pair clearance and >=10 mm boundary margin;
  numerical geometry tolerance 0.1 mm. No intersection area >1e-12 m².
- Locked objects retain exact poses. Movement label: displacement >3 mm OR
  major-axis angular error >3°. No-op locks all objects, has exact source-target
  equality, false movement labels and empty row/relation supervision.
- Keep-all source must already satisfy final geometry. Failure is provably
  infeasible. Repeated no-op must also satisfy the requested layout predicates.
- Full/left/right requested regions: [0.02,0.02,0.58,0.38],
  [0.02,0.02,0.28,0.38], [0.32,0.02,0.58,0.38] m.
- Rows run left-to-right then top-to-bottom. Sort by a*b ascending/descending;
  validator allows at most 5% size-order inversion tolerance. Alignment <=5°.
- Infer rows from target coordinates, with max y span 10 mm. Horizontal/vertical
  edge gaps must be requested 15/20/30 mm ±5 mm (+0.1 mm numerical tolerance).
  Separate color bands use 2× the requested vertical gap. Every row is color-pure
  and each color occupies one consecutive run of rows. Color order itself is free.
- Partial semantic predicates apply to selected objects; all-scene safety and
  immutable locks apply to every object. Unselected objects need not form rows.

The legacy Teacher groups colors independently, chunks each group with K=1..4, centers each
row, and tries fewest rows, bounding area, width, then lexicographic positions.
Within a group, ties use dimensions, color and observable source coordinates.
Partial templates translate on a 5 mm lattice, nearest region center first,
then upper before lower and left before right. Search budget is 20,000 candidates.
Exhausting templates or budget is `generation_failure`, never `infeasible`.
An independent validator disagreement raises an error and stops generation.

The active balanced Teacher uses K=1..N_selected, sequential and balanced partitions,
J_shape for Full and global J_move-first lattice ranking for Partial. This Teacher
and its targets are unchanged by the English-only conversion.

## Identity, data flow and splits

Object IDs belong to observed source instances and are copied into targets.
There is no target Hungarian matching or reassignment. Instance ID i+1 maps to
the ith source object in render metadata; annotations keep both IDs. Target
poses, current poses, dimensions and semantic selection/lock sets are serialized.
Detection boxes and normalized centers are auxiliary labels, not model inputs.
The dataset package is independent of neural training. The separate
`knolling_model` package now implements the S0 model, matching/loss, evaluator
and launcher; see [S0 implementation](s0-implementation.md). No training or robot
execution has been performed.

The active bounded grammar emits English text from structured semantic fields.
Selection references all objects, a visible color, or an unambiguous leftmost
object. Raw IDs never appear in model-input text. Colors, directions, requested
gaps, region, size-order convention and lock behavior are stated explicitly.
`preference_text` is reserved and empty; preferences here are expressed directly
in the instruction. Free-form parsing, dialogue history and unsupported arbitrary
instructions are outside this pilot. Unsupported/ambiguous status interfaces
have tests; the rendered pilot contains feasible positive samples only.

The English conversion changes only instruction text/language; it never calls
Blender or recomputes target layouts. Parent annotations are hash-verified archival
evidence, excluded from active split indexes. Every active split is checked against
the manifest, and every instruction must match the English semantic renderer.
Original bilingual grammar remains available only through explicit legacy APIs.
User-facing examples and language evaluation are English; multilingual support is
optional future work. Neural architecture and the current text encoder stay fixed.

Scene families are assigned before rendering. Paraphrases/counterfactual tasks
must retain their family. The current pilot has two semantically distinct tasks
for 30% of families; it does not claim broad paraphrase generalization. Geometry
families normalize the planar footprint by the major length and ignore color,
uniform scale **and height**. Thus a changed height cannot move an otherwise
identical top-down shape into another split. Shape classes span splits, geometry
families do not. Every pool has 32 families. Registry generation is fixed seed;
no external mesh license, download or preexisting code is involved.

Audit independently checks scene/asset families, normalized geometry aliases,
identical physical scenes under new IDs, relation labels, unit conversions,
manifest counts and file hashes. RGB/mask checks reproject actual polygons using
pixel centers; per-object IoU >=0.98, centroid <=1 px, interior color agreement
>=99%, white clipping <=1%. A re-audit removes stale COMPLETE before starting.

## Reproducibility and limitations

Python dependencies are pinned in requirements-lock.txt. Blender is 5.1.0,
Cycles CPU 32 samples, fixed per-family seed. Dataset sampling is deterministic
without changing global RNG. Exact geometry/annotation hashes and repeated
PNG bytes are tested. PNG render-time stamps are stripped without changing RGB.
Cross-platform/cross-Blender bitwise reproducibility is not promised.

Teacher acceptance/rejection statistics count every primary proposal and list
failure categories. Rejections are not mislabeled as negative training examples.
The 100-scene run is a quality pilot, not a performance benchmark. Partial cases
lock a subset of safe scattered objects; multi-turn sequences are not generated.
Before scaling: human QC, richer language/asset distribution, resumable rendering,
streamed jobs and throughput/storage planning. The current batch is intentionally
small, synchronous and all-or-fail; it does not resume partial runs automatically.
The CLI enforces a 100-family cap until a larger run is explicitly authorized.
