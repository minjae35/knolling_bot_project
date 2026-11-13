# Final English v1.3 dataset quality review

2026-11-03 · Reviewed code `758e5ef` · Dataset `artifacts/pilot_v1_3_en/`

**Recommendation: retain the current dataset and proceed to S0 after the user's
human review gate.** No Critical defect or clear systematic Major defect was
confirmed. Another Teacher revision or wholesale wording migration is not
justified before the first overfit experiment. This is a technical recommendation,
not human approval or evidence that the Neural Network will train successfully.

## Coverage and disposition

| Result | Count |
|---|---:|
| Instructions and source → target pairs reviewed by the assistant | 130 / 130 |
| Distinct complete English strings read, with all sample IDs mapped | 81 |
| Corresponding v1.2 targets compared | 130 |
| Technically suitable for S0, assistant judgment | 130 |
| Critical / Major sample defects | 0 / 0 |
| Samples with Minor notes | 124 |
| Samples with no individual issue, classified Acceptable | 6 |
| Samples with layout-preference notes, included in the 124 | 25 |
| Corrected samples / changed poses / re-rendered views | 0 / 0 / 0 |
| Human-approved samples | 0; review records remain pending |

“Flagged” and “suitable for S0” overlap: a Minor style/preference note does not
reject a sample. Six Keep-all instructions are concise and unproblematic. Every
other instruction contains the repeated technical template and is tagged L1.
Dataset-level coverage limitations are reported separately, not disguised as
incorrect individual targets.

The assistant inspected all 13 ten-sample boards, including sources, v1.2 and
v1.3 targets, and read all complete distinct strings with their sample mappings.
Eight borderline cases were additionally inspected at original RGB resolution:
`scene_0024_primary`, `scene_0028_primary`, `scene_0044_primary`,
`scene_0079_primary`, `scene_0068_primary`, `scene_0088_primary`,
`scene_0089_primary`, `scene_0095_primary`. Visual ratings below are assistant
judgments, not automated aesthetic measurements or user sign-off.

## English findings

**L1 — Minor, 124 samples:** understandable and semantically consistent, but
written like a technical contract rather than everyday instructions. Lengths are
62 words (41 samples), 73 (16), 78 (53), and 89 (14); the six Keep-all strings
have nine words. Median length is 78. “Long axes horizontal” is understandable
elliptical wording, but “with their long sides horizontal” is more natural.
The repeated row-band definition and inter-group explanations add substantial
boilerplate. Singleton instructions awkwardly request rows, ordering and gaps
that have no pairwise effect in that scene. No wrong number, direction, reference
or lock instruction was found. Retain for S0; do not confuse verbosity with bad GT.

Affected IDs are **every sample except** `scene_0075_primary`,
`scene_0078_primary`, `scene_0082_primary`, `scene_0085_primary`,
`scene_0092_primary`, `scene_0096_primary`. The local per-sample CSV lists all 124
explicitly, with the complete instruction available in the comparison page.

**Language diversity — insufficient for natural-language generalization, not an
S0 blocker.** The 81 unique strings arise from slot substitutions in one main
grammar plus Keep-all, not 81 paraphrase styles. Colors, regions, axes and size
orders vary, but sentence structure barely does. The 30 diagnostic counterfactual
pairs test different structured requests, not robustness to natural paraphrases.
Do not claim free-form English competence from this dataset. Add a small,
versioned compact/paraphrase set only when evaluating that capability after S0.

**RGB + instruction grounding:** all selected sets have supported visible color
or current-source leftmost references; no hidden ID appears in the instruction.
Objects, colors and orientations are legible in the reviewed views. The fixed
calibrated workspace supplies metric scale. Bounding-rectangle area, region
bounds, gap conventions and deterministic target choice are learned task
conventions, not information an arbitrary photograph intrinsically provides.
There are no conflicting targets for an identical RGB/string pair in this pilot.
The ordinary words still allow several valid layouts; Teacher-coordinate
agreement and task success remain separate metrics.

[English arrangement conventions](english-instruction-defaults.md) documents
those existing defaults and three shorter wording proposals. These proposals are
**not activated** and do not replace active annotations. Removing “row band” or
changing “bounding-rectangle area” to “size” without such a contract would create
real ambiguity; simply deleting those phrases is not recommended.

## Layout findings and root causes

| Finding | Classification | Evidence and action |
|---|---|---|
| F1: optional singleton rows in Full | Minor, 14 samples | A feasible current-policy candidate has fewer singleton rows, but loses on J_shape. All inspected results remain aligned and meaningfully organized. Retain. |
| F2: unequal ungrouped row counts | Minor, 2 samples | `scene_0018_counterfactual`: 3+3 → 4+2; `scene_0048_primary`: 4+4 → 5+3. Different physical dimensions and workspace aspect ratio explain the score choice. Retain. |
| P2: singleton rows in multi-object Partial | Minor, 9 samples | Minimum movement is prioritized over shape; locked context constrains placement. Selected blocks are ordered and preserve all locks. Retain. |
| G1: large color-group gaps | Acceptable, 8 samples | 30 mm within groups and 60 mm between groups are explicitly requested. Sparse-looking output is not an accidental Teacher spacing error. Retain. |
| Different-size objects, unequal row widths | Acceptable | Common row centers, bounding-envelope gaps and area order are consistent. Do not require equal-width columns or falsely sort by visible silhouette area. |
| Color-forced singleton rows | Acceptable | A color represented by one object must occupy its own row under the current instruction. Mixing colors to avoid a singleton would be incorrect. |

F1 IDs (all `_primary`): `scene_0000`, `scene_0007`, `scene_0015`, `scene_0022`,
`scene_0024`, `scene_0028`, `scene_0043`, `scene_0044`, `scene_0045`, `scene_0057`,
`scene_0079`, `scene_0089`, `scene_0095`, `scene_0099`.

P2 IDs (all `_primary`): `scene_0005`, `scene_0033`, `scene_0034`, `scene_0039`,
`scene_0064`, `scene_0068`, `scene_0069`, `scene_0081`, `scene_0088`.

G1 IDs: `scene_0025_primary`, `scene_0025_counterfactual`,
`scene_0046_primary`, `scene_0046_counterfactual`, `scene_0059_primary`,
`scene_0066_primary`, `scene_0076_primary`, `scene_0089_primary`.

### What J_shape actually prefers

The score is `max(W / region_width, H / region_height)`. It is not an aesthetic,
row-balance, singleton-count or empty-space objective. Area and row count only
break ties after this score, so even a small improvement can change the layout.
Nine of the 80 Full samples use all-singleton rows despite a feasible alternative
with fewer singletons. All nine request a narrow left/right region. This is a
systematic preference for lower normalized maximum extent, but the visual review
does **not** establish a systematic Major training defect.

| Sample | v1.2 rows → v1.3 rows | J_shape v1.2 → v1.3 | Judgment |
|---|---|---|---|
| `scene_0000_primary` | 2 → 1+1 | 0.661538 → 0.307692 | Compact horizontal objects stacked in the right region; useful simple organization. |
| `scene_0024_primary` | 3+2 → 2+2+1 | 0.451786 → 0.441667 | Small score gain creates a bottom singleton; cosmetic preference. |
| `scene_0028_primary` | 2 → 1+1 | 0.442308 → 0.433333 | Either arrangement is reasonable; no semantic reason to relabel. |
| `scene_0044_primary` | 2+2 → 1+1+1+1 | 0.696154 → 0.616667 | Clear column; grid may be preferred subjectively. |
| `scene_0079_primary` | 2+2 → 1+1+1+1 | 0.584615 → 0.580556 | Strongest near-tie preference example; still orderly, not incorrect. |
| `scene_0089_primary` | 2+3+2 → 2+2+2+1 | 0.992308 → 0.741667 | Narrower blue block with a singleton; requested 60 mm group gap remains. |
| `scene_0095_primary` | 3+1+2 → 2+1+1+2 | 0.869231 → 0.633333 | Splits the red group; colors remain pure and contiguous. |

All 80 Full candidate sets were inspected numerically for feasible
fewer-singleton alternatives; the complete values are in `full_candidates.json`.
These diagnostic alternatives were never installed as targets or rendered.
Adding a singleton penalty, score tolerance or new objective would encode a
subjective preference and change supervision. Defer unless the user rejects these
layouts or S0 reveals a concrete label-learning problem.

### Partial and No-op usefulness

All 30 Partial samples satisfy their requests; all selected objects physically
move or rotate, and all locked poses are unchanged. There are 11 one-object,
11 two-object, two three-object and six four-object selections. The 11 singleton
IDs (all `_primary`) are `scene_0004`, `scene_0009`, `scene_0029`, `scene_0037`,
`scene_0042`, `scene_0058`, `scene_0063`, `scene_0074`, `scene_0084`, `scene_0086`,
`scene_0098`. Classify them as **Acceptable grounding/keep supervision**, with the
Minor wording issue above, not evidence of relational knolling.

The other 19 examples teach local multi-object arrangement. None starts from a
documented organized whole scene, and there are no multi-turn episodes. This is
a coverage limitation for interactive refinement, not an invalid-label problem.
Do not move scattered locked objects merely to improve global appearance.
Mean squared selected-center displacement falls from 0.03963919 to 0.01629424 m²
relative to v1.2, with no increase in any Partial sample. Lower movement does not
prove better aesthetics. For example, `scene_0068_primary` changes from one row
to a column while reducing movement cost from 0.06633985 to 0.04552444 m².

All 20 No-op targets exactly preserve source poses: 14 repeated arrangements
already satisfy the task and six explicitly request Keep-all. The six scattered
Keep-all scenes are correct preservation examples. They must not be judged as
failed attempts to tidy a scene, nor rearranged to improve J_shape.

## Validation, deliverables and next gate

All three dataset audits pass read-only: v1.2, original v1.3 and English v1.3,
130 samples each. The English audit includes geometry, boundaries, collisions,
identities, instructions, deterministic policy reproduction, render projections,
language lineage and split integrity. Independent comparisons found zero
semantic inconsistencies, conflicting identical RGB/string labels, or No-op pose
changes. The complete suite passed **78 tests in 173.47 seconds, zero failures
and zero skips**, with Blender, both builders, both migrations and all three
actual dataset acceptance tests enabled:

```sh
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 RUN_MIGRATION_TESTS=1 RUN_LANGUAGE_MIGRATION_TESTS=1 \
PILOT_DATASET=artifacts/pilot_v1_2 PILOT_V13_DATASET=artifacts/pilot_v1_3 \
PILOT_EN_DATASET=artifacts/pilot_v1_3_en \
/Users/minjae/miniconda3/envs/knolling/bin/python -m pytest -q
```

Temporary renders made by regression tests do not alter any delivered dataset.
Post-test SHA-256 tree comparisons confirm all **1,041 v1.2 files, 1,614 original
v1.3 files and 1,750 English v1.3 files** are unchanged against snapshots taken
before this review. This includes annotations, images, manifests, QC reports,
metadata and pending human-review records. Review-page validation also confirms
130/124 unique cards on the all-sample/flagged pages and valid local image/link
targets. Results are saved in `artifacts/final_quality_review/validation.json`.
The English manifest remains
`8000428a04522f50cf37567d85d442ac94d1f8a87121747d4243e33974320d06`.

Local review materials, outside all three immutable datasets:

- `artifacts/final_quality_review/index.html`: all 130 comparisons, complete
  instructions, per-sample classifications, evidence, root causes and actions.
- `flagged.html`: all 124 flagged samples; use the layout filter for the 25
  subjective layout cases. Raw RGB and existing movement reports are linked.
- `sample_reviews.csv` / `sample_reviews.jsonl`: explicit IDs and separate
  automated facts / assistant judgments; no human approvals.
- `metrics.json`, `full_candidates.json`, three `*_audit.json` files,
  `review_summary.json`, 13 boards and two raw comparison sheets.

**S0 decision: technical GO, human approval still pending.** There are 38 Train
samples with 2–4 objects (18 Full, 10 Partial, 10 No-op), enough to select the
specified 32-sample overfit set without borrowing from held-out splits. Use only
the active manifest and original RGB, with fixed canonical English initially.
Do not infer generalization from memorization. The existing human quality gate
and `training_approved.jsonl` have not been bypassed or marked complete.

The user's remaining subjective choices are whether to accept the flagged
columns/row splits for the initial pilot and whether technical wording is
acceptable for that diagnostic. My recommendation is yes to both. Shorter user-like
English and organized-source refinement should be tested later; they need not
delay this implementation diagnostic. No architecture change, new dataset,
Teacher revision, training implementation or training run was performed.
