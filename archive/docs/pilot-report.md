# Phase 1 pilot completion report

2026-11-03. M0–M8 were implemented from scratch in the new repository
`minjae35/knolling_bot_project`. No code from the previous project was reused.
No neural network, training, robot simulation, or 50,000-scene production
dataset was produced.

## Results

- **100** distinct scene families, **130** samples, and **210** real Blender render views.
- Primary: full 50 / partial 30 / no-op 20. No-op is 14 repeated instructions and 6 explicit keep-all.
- Thirty families have an extra sample that shares the input image but uses a different instruction and answer.
- 495 source objects, 2–8 per scene. Box 166 / chamfer 188 / capsule 141.
- Colors: red 168 / green 150 / blue 177. 114 asset families were used.
- Instruction language: Korean 73 / English 57 samples.
- Every sample passed identity preservation, ID, geometry, lock, applicable instruction checks, and the render check.
- Audit errors: **0**. 760 files were checked, including annotations and image, mask, and render metadata.

Primary proposals accepted 100 of 101: acceptance **99.01%**, exclusion **0.99%**.
The one exclusion has reason `candidate_family_exhausted` and status
`generation_failure`. It was not treated as evidence that the instruction is
impossible. Extra instructions accepted 30 of 30. Zero samples were excluded
by the render check. These figures are the **teacher's data-generation
acceptance rate**. They are not a future neural-network or robot success rate.

| Check | Measured | Pass criterion |
|---|---:|---:|
| Target minimum envelope gap | 10.1688 mm | 10 mm, numerical tolerance 0.1 mm |
| Target minimum boundary margin | 12.5813 mm | 10 mm, numerical tolerance 0.1 mm |
| Minimum mask-to-true-footprint IoU | 0.998774 | ≥0.98 |
| Maximum center projection error | 0.636553 px | ≤1 px |
| Minimum interior RGB color agreement | 100% | ≥99% |

| Split | Scene families | Samples | Asset families used |
|---|---:|---:|---:|
| Train | 70 | 89 | 32 |
| Validation | 10 | 13 | 26 |
| Development | 10 | 13 | 28 |
| Test | 10 | 15 | 28 |

Scene family, asset family, normalized planar shape, and the hash of the same
scene with IDs removed were checked. There is no overlap across splits.
Changing only color, uniform scale, or height does not create another asset
family. This is a small **geometry-OOD pilot**. The split sizes do not mean
that model performance can be estimated with high confidence.

## Implementation and verification history

Each row is the measurement at that time. On the final code, **all 39 tests,
including the real pilot check, passed together in 48.05 seconds**. The ordinary
fast tests skip Blender, the full builder, and the real pilot check depending
on environment variables, so those runs are counted separately.

| Module / change | Commit | Conventional commit | Verification |
|---|---|---|---|
| M0 | `0e9c66f` | `feat(dataset): add schemas and coordinate utilities` | 3 passed |
| M5 | `82e60cb` | `feat(dataset): validate geometry and layouts` | 13 passed |
| M1 | `6b9b869` | `feat(dataset): implement procedural assets` | 18 passed |
| M2 | `a3e7ddc` | `feat(dataset): generate randomized scenes` | 19 passed, including a 100-seed check |
| M3 | `99e8151` | `feat(dataset): generate language instructions` | 26 passed |
| M4 | `f091444` | `feat(dataset): implement layout teacher` | 30 passed; repeated no-op distinguished from search failure |
| M6 | `f372f76` | `feat(dataset): add Blender rendering` | geometry and RGB checks passed; an extra file-hash check found a metadata difference |
| M6 fix | `9fda1c3` | `fix(dataset): canonicalize Blender PNG metadata` | 32 passed, including a real repeated render |
| M7 | `9c4960e` | `feat(dataset): build annotated samples` | 34 passed, including a real 10-scene integrated generation |
| M8 | `47ca812` | `feat(dataset): enforce split integrity` | 38 passed, including real renders and the builder |
| Language fix | `8cda9be` | `fix(dataset): correct Korean selection particles` | fast check 36 passed; 3 external data/render checks run separately |
| Final pilot check | committed with this report | `test(dataset): validate pilot dataset` | all 39 passed, including a full re-audit of the 100 scenes and annotation regeneration |

Every change is pushed to `main` in order. There is no force push and no
history rewrite. `f372f76` was pushed before the extra M6 hash check had been
read. The pixels were identical; only the PNG creation time and render time
differed. A later fix stripped that metadata and checked file hashes of real
repeated renders. The failure was not hidden and the history was not erased.

The particle error in Korean partial instructions was fixed in `8cda9be`.
Only the 18 affected pilot sentences were updated. Every semantic field,
coordinate, object, and image hash was confirmed identical before and after.
The annotation hash, manifest, and QC report were updated together and the
full set was re-audited. `provenance.json` records both the render revision
and the post-fix annotation revision.

## Outputs and how to review them

Local path: `artifacts/pilot_v1_2/` (about 77.65 MB including QC; not in Git).

```text
asset_registry.json                 procedural asset registry
generation.json                     proposal, acceptance, and exclusion history and distributions
provenance.json                     seed, tool and coordinate info, code revisions
manifest.jsonl                      paths and SHA-256 of the 130 annotations
{train,validation,development,test}.jsonl
audit.json, COMPLETE.json           final audit and the manifest hash
scenes/<family>/source/             RGB, uint16 instance mask, render metadata
samples/<sample>/annotation.json    current and target pose, IDs, instruction, relations, validation
samples/<sample>/target/            target RGB, mask, and metadata, except for no-op
qc/index.html                       offline QC that can show every sample
qc/contact_sheet.png                nine representative input and target scenes
qc/summary.json                     automatic QC numbers
work/                               render jobs and Blender logs for reproduction
```

Open `qc/index.html` in a browser and use the mode and split filters. Each
card has the Korean or English instruction, the input and target images,
consistent object IDs, and an `[L]` lock mark. It also links to the original
annotation and mask. Mask values are the integers 0–8, so a normal image
viewer can show them as black. A no-op target reuses the exact source render.

The assistant checked the nine pairs on the representative contact sheet, the
8-object layout of `scene_0006_primary`, the keep-all of `scene_0075_primary`,
and the changed instruction of `scene_0000_counterfactual`. In that sample,
no missing object, visual collision, or missing ID or lock mark was observed.
The Korean particle error was found and fixed during this review.
**A person has not reviewed all 130 samples.** Review of the user's natural
language intent, readability, and layout diversity is still
`pending_user_review`. An automatic pass was not recorded as a human approval.
Keep-all preserves an already safe scene, so missing axis alignment is expected.

Final manifest SHA-256:
`b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27`

## Re-verification and what is next

```sh
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 PILOT_DATASET=artifacts/pilot_v1_2 conda run -n knolling python -m pytest -q
```

**Small-scale functional and consistency checks of the dataset generator are
done.** That does not mean the system is ready to produce 50,000 scenes. The
current CLI allows at most 100 families. Rendering is synchronous in small
batches and cannot resume from an interruption. Before a large run, the user
still needs visual and language QC, a chunk and resume design, and a check of
throughput and storage. The current distribution is three procedural shapes,
three colors, a fixed bird's-eye view, and a limited Korean-and-English
grammar. It does not guarantee free-form instructions or real-asset
generalization. Partial is a single step that edits part of a safe scattered
scene. There is no multi-turn dialogue history. The rendered data does not
include non-executable ambiguous, unsupported, or infeasible instructions.
Only the status interfaces for those cases were tested.

Implementation and generation stop at this point. The production dataset was
not scaled up, and the neural pilot, training, and robot simulation were not
run.
