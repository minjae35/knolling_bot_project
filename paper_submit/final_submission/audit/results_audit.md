# Results and data audit for the course paper

This audit read existing code, stored annotations, frozen score rows, and experiment summaries. It did not retrain a model, invoke inference, or rescore the sealed holdout. All new files are under `paper_submit/`.

## Primary result: independently reaggregated

Original score rows were located at `/tmp/v1_holdout.jsonl` and `/tmp/v2_holdout.jsonl`. They are preserved byte-for-byte under `audit/evidence/`, with source paths and SHA-256 in `audit/evidence/provenance.json`. `scripts/audit_results.py` reproduces `audit/verified_results.json` using the Python standard library. It verifies 640 unique matching sample IDs, 128 families with five samples each, matching mode/count/family metadata, and the `v2hold_` prefix. These aggregate results exactly match the release record.

| Mode | v1 success | v1 (%) | v2 success | v2 (%) | Gain (percentage points) |
|---|---:|---:|---:|---:|---:|
| Full | 287/384 | 74.74 | 311/384 | 80.99 | 6.25 |
| Partial | 84/127 | 66.14 | 104/127 | 81.89 | 15.75 |
| No-op | 118/129 | 91.47 | 125/129 | 96.90 | 5.43 |
| Overall | 489/640 | 76.41 | 540/640 | 84.38 | 7.97 |

The paired outcome table is 489 shared successes, 51 new successes, zero lost successes, and 100 shared failures. Of the 128 families, 29 improve, zero worsen, and 99 tie. A paired family bootstrap samples 128 families with replacement, retaining all five correlated instructions per sampled family. Using Python `random.Random(20261112)`, 10,000 resamples, sorted-draw indices 250 and 9749, the 95% interval for the gain is **[4.84375, 11.5625] percentage points**, rounded **[4.8, 11.6]**. Do not bootstrap the 640 rows as independent images: there are only 128 distinct RGB source images. The estimate conditions on the two fixed checkpoints; it does not capture training-seed variance.

Absolute failure reduction is 51/151 = 33.77%. This is optional; percentage-point gain is clearer and should be primary. No independent runs or seed robustness were measured.

| Objects | v1 | v2 |
|---:|---:|---:|
| 2 | 91/95 | 91/95 |
| 3 | 86/95 | 88/95 |
| 4 | 64/90 | 75/90 |
| 5 | 77/90 | 79/90 |
| 6 | 65/90 | 72/90 |
| 7 | 59/90 | 70/90 |
| 8 | 47/90 | 65/90 |

This is all modes, not full-only. For full-only eight-object scenes, v1 is 28/54 and v2 is 39/54. Results are not monotonic in object count, and the distinct family sample sizes are only 18–19 per count. Describe the eight-object gain as an observed subgroup difference, not as a formal interaction effect.

## Error diagnosis

| Error tag | v1 | v2 |
|---|---:|---:|
| Horizontal gap | 51 | 49 |
| Vertical gap | 48 | 42 |
| Object count | 45 | 5 |
| Requested region | 17 | 4 |
| Size order | 13 | 11 |
| Keep pose | 9 | 4 |
| Clearance | 9 | 7 |
| Mixed-color row | 2 | 0 |
| Collision | 1 | 0 |
| Boundary | 1 | 0 |

Tags overlap. **71 of the 100 remaining v2 failures have at least one horizontal/vertical gap error**, not 91 distinct spacing failures. Count mismatches short-circuit scoring, so fewer count failures expose subsequent errors; raw per-tag reductions are not isolated causal ablations. Of the 51 recovered scenes, 27 previously failed only the count check. The final v1/v2 comparison combines multiple changes. Avoid attributing all +8 points to constrained decoding or any single mechanism without an ablation.

Recorded per-scene latency median is 0.13148 s for v1 and 0.13721 s for v2 (standard even-length median); means are 0.16924 and 0.19752 s. The older comparison script used the upper middle element, reporting 0.13161 and 0.13723 s. This explains the release's 0.132/0.137 s rounding. These logs are not a hardware-controlled speed benchmark; recommend omitting latency from the main claim.

## Dataset and leakage checks

`audit/dataset_overlap.json` is a fresh read-only aggregation of `artifacts/generalization_20000/dataset/annotations.json` and all `artifacts/v2_independent_eval/families/*/family.json`, using the preserved `audit/evidence/audit_holdout.py` script. The six dataset hashes in `audit/dataset_hashes.json` exactly match `docs/v2.0.0.md`.

- Generated corpus: 20,000 instruction/scene records, 4,000 image families; 16,000 train, 2,000 validation, 2,000 test. Object-count family frequencies are 572 at counts 2–4 and 571 at 5–8. Modes: 12,000 full, 3,972 partial, 4,028 no-op. A sampled partial command can become no-op, explaining unequal partial/no-op totals.
- Final holdout: 640 records, 128 source-image families; 384 full, 127 partial, 129 no-op. Object-count family frequencies: 19 at counts 2–3 and 18 at counts 4–8. Holdout sources contain 635 total objects.
- Intersections across corpus versus final holdout: **zero scene-family IDs, zero recorded source-RGB hashes, zero asset-family IDs**. Holdout uses 64 development asset families; corpus uses 192 asset families. This is held-out procedural asset-family evaluation, not natural-image generalization.
- Holdout shape mix: box 201/635 (31.65%), capsule 156/635 (24.57%), chamfer 278/635 (43.78%). Corpus: box 6,555/19,994 (32.78%), capsule 6,354/19,994 (31.78%), chamfer 7,085/19,994 (35.44%). Thus the holdout is somewhat shifted in procedural shape mix.
- The official holdout contract has `scored=true`, `scored_once="v1.0.0 and v2.0.0-rc1, not for model selection"`. Its manifest retains the earlier pre-scoring `scored=false` snapshot; the separate contract is the later state. This is not grounds to repeat scoring.
- Split implementation is `src/knolling_model/experiments/expanded_data.py:32`: `(i//20)%10`, buckets 0–7 train, 8 validation, 9 test, assigned before rendering. Therefore related instructions share splits.
- `/tmp/make_v2_holdout.py` exists on this machine, although not in git. It should not be invoked. Public-source reproducibility remains incomplete, as accurately stated in project docs.

## Evaluation exactness

`src/knolling_model/arrangement/evaluation_v3.py:18` defines `semantic_errors_v3` with keep position 0.001 m, keep angle 1 degree, axis 2 degrees, gap 0.002 m, and row alignment 0.002 m. `src/knolling_model/evaluation.py:124` calls `geometry(positioned, 'target_pose', .010, .010)` for 10 mm clearance and boundary margin. Evaluation compares conservative oriented rectangles; visual mesh silhouettes need not fill those rectangles. A successful scene satisfies all checks after count/identity matching, not merely small average pose error. The result is synthetic target-pose accuracy, not motion execution.

## Historical development evidence suitable for an appendix

Preserved JSON summaries support the following. These are **different development subsets and historical system states**, not rows in one controlled final-system ablation table.

1. `artifacts/experiments/end_to_end_joint/result.json` records `status=FAIL`, `stop=collapse`, best update 0, no test use, no adoption. `history.jsonl` has baseline and update 200: development end-to-end success 2/38 to 0/38; validation 24/152 to 1/152; oracle and single checks each drop to zero. This documents an attempted jointly optimized path and explains a staged design choice; it does not establish that end-to-end training is generally inferior.
2. `artifacts/experiments/joint_stage/result.json` records an adopted restricted joint update: best update 30, LR 1e-5, trainable size refinement / row step / final row-origin layer. Frozen RT-DETR, physical heads, language, goal, relation, row assignment, row order, row angle. RGB validation improves **42/152 to 54/152**; median horizontal error 0.858 to 0.784 mm and median vertical error 2.015 to 1.588 mm. Six of 120 attempted scenes excluded. It did not train over all 9,600 full training records.
3. `artifacts/experiments/selector_full/result.json` records eight epochs, 1,272 updates, 1,792 fit families, 5,376 candidate full records, 5,085 usable records; 249 alignment and 42 count skips. Validation full scenes: **222/1,200 to 235/1,200**, chosen at epoch 1. Row accuracy 68.52% to 75.86%; slot accuracy 61.87% to 71.77%. Epoch 8 has higher row accuracy (84.88%) but 233/1,200 task passes, illustrating mismatch between component and task metrics. Dev256: 79 to 88, 14 gains and five losses. Dev512: 146 to 162, 26 gains and ten losses. These are earlier component validation scores, not the current 84.4% system.
4. `artifacts/experiments/selector_full/axis_alignment.json`: validation **235/1,200 to 285/1,200**, 50 gains, zero losses; axis-tag failures 370 to zero. `vertical_gap_confirm.json`: validation **285/1,200 to 423/1,200**, 140 gains and two losses; vertical-gap tags 535 to 188. Dev512 162 to 259 (97 gains, no losses), Dev256 88 to 141 (53 gains, no losses). These records can support a sequential development study, with clearly labeled changing decoder versions.

**Critical naming trap:** selector diagnostic JSON has a `holdout` of 640 one-full-scene-per-training-family examples. `experiments/end_to_end/dev_splits.py:17` and `:25` show this is the one-fifth slice of 3,200 training families. It is not the independent final `v2hold_` set of 128 families and five commands each. Older “holdout” slices also have historical exposure recorded in `end_to_end_joint/result.json`. Keep all of those out of the final-test claim.

## Qualitative evidence with honest provenance

### Recommended final-system examples

- `audit/evidence/full8b_choice.json`: eight-object full/color-group/right-region success, sample `expanded_717287879ce16878c5558650_4`. It contains the complete `scene.objects` records with `asset.footprint`, `source_pose`, and **predicted** `target_pose`. Source RGB: `artifacts/generalization_20000/dataset/families/2967/source/rgb.png`. Saved rendered prediction: `docs/v2.0.0/demo-gallery/renders/expanded_717287879ce16878c5558650_4/predicted/rgb.png`. This is the data behind gallery figure 13.
- `audit/evidence/full67_choice.json`: six-object full horizontal success, sample `expanded_e2ee7a75ae3a34826b36d71f_0`, same schema, source family directory `0165`, gallery figure 11. `full7_choice.json` is seven objects, family directory `0166`, gallery 12.
- `audit/evidence/v2_validation_smoke.json`: one eight-object family under full, partial and no-op commands, with predicted poses and outcome fields. Sample family `expanded_09aa3f57968449ce8a09f291`: full fails vertical gap; partial selects correct blue objects but fails clearance; no-op passes. `objects` records use `object_id`, `x_m`, `y_m`, `theta_rad` for predicted targets, plus detected poses. Join object IDs to `audit/evidence/v2_smoke_source_records.json` for original source poses and mesh footprints. Source RGB is `artifacts/generalization_20000/dataset/families/0160/source/rgb.png`.
- The preserved smoke script explicitly calls `load_adopted` and `poses_from_detection`, uses only the first supported validation scene of each mode, and writes these results. Its outcomes are also recorded in `docs/reproducibility.md`. These are selected illustration cases, not an estimate of the success rate. The choice files and smoke record do not contain checkpoint hashes or commit metadata, so describe their provenance as saved adopted-system validation examples, not a new release-level reproducibility test.
- Raw holdout JSONL has success/error/timing but **no poses**. Do not create fake holdout target plots from these records or use GT targets as predictions.

### Historical gallery warning

`docs/v2.0.0/demo-gallery/summary.json:2` explicitly records commit `26dca53` (v1). Its first ten illustrations must not be presented as v2 output. They can be shown as historical examples with that label, but later v2 smoke/choice records are preferable. `/tmp/v2_val.jsonl` also predates final adopted behavior (the first family has count failures whereas the later smoke detects all eight), so its filename alone is insufficient version provenance.

The plotted shape/appearance in predicted target visualizations may use ground-truth source meshes to render matched objects; this is a **visualization-only** operation, not inference access to target geometry. Label schematic pose visualizations and note this convention. Arrows represent source-to-target displacement, not collision-free trajectories.

## Assignment guidance search

No rubric, syllabus, assignment, grading or course document was found in the repository and a filename search of `/workspace` up to depth four. No professor/course/author identity should be invented. The parent is handling any connected-source search and user clarification. In the absence of a rubric, emphasize clear research question, accurate method, evidence-backed results, controlled claim scope, limitations, and reproducibility rather than claiming compliance with unknown grading criteria.
