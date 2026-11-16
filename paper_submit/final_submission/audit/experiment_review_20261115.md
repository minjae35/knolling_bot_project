# Independent experiment review — 15 November 2026

This review inspected saved records, version-controlled evaluator/loading code, dataset annotations, checkpoint bytes, and the current manuscript. It performed no model inference, training, parameter selection, or dataset modification. All new output is confined to this audit directory.

## Verdict

The main numerical claims and current limitations are supported. No substantive numerical correction is required. The existing manuscript correctly distinguishes the final paired release comparison from historical development diagnostics, inference from physical execution, and the end-to-end objective from adopted staged learning.

Machine-readable independent checks are in `experiment_independent_checks_20261115.json`.

## Independently reproduced results

- Both frozen logs contain 640 unique, identical sample IDs, with 128 families and exactly five records per family. Every paired record agrees on family, task mode, and object count.
- Task flags agree with empty/nonempty error lists for all 1,280 records.
- Paired outcomes are 489 both-pass, 51 v2-only pass, zero v1-only pass, and 100 both-fail.
- The gain is 51/640 = 7.96875 percentage points. A new standard-library recomputation with the documented seed, 10,000 family resamples, and published percentile indices gives [4.84375, 11.5625] points.
- Family gain counts are 99 ties, 20 families gaining one command, three gaining two, two gaining three, one gaining four, and three gaining all five. Thus 29 improve and none worsen.
- Remaining failures partition into 29 horizontal-gap only, 22 vertical-gap only, 20 both spacing tags, and 29 with neither spacing tag. The 71/100 spacing statement is correct.
- All 27 frozen-manifest hashes, all 15 original-to-archive evidence hashes, all six recorded dataset hashes, and all eight checkpoint/config hashes listed in the release record match the actual current files.

## Dataset separation

A fresh scan of the actual annotations, independent from the existing overlap JSON, confirms:

| Pool | Pairs | Source families / RGB hashes | Asset families |
|---|---:|---:|---:|
| Train | 16,000 | 3,200 | 64 |
| Validation | 2,000 | 400 | 64 |
| Official test | 2,000 | 400 | 64 |
| Separate final set | 640 | 128 | 64 |

Every pairwise intersection between train, validation, and official test is zero for source-family IDs, recorded RGB hashes, and asset-family IDs. The separate final set also intersects none of those pools under these three keys.

The source generator selects split-specific registries before generating scenes: `src/knolling_model/experiments/expanded_data.py:32–46`. Therefore the appendix could optionally state that the 20,000-pair corpus uses disjoint asset-family pools as well as disjoint source-image families. This would be a supported strengthening, although it is not necessary.

Important scope: this proves disjointness relative to the 20,000-pair corpus. The final set uses the registry's development assets; it does not by itself establish asset novelty relative to every earlier historical training corpus. Keep the current narrower wording and do not call the final assets globally unseen.

## Release and scorer comparability

A Git comparison of v1.0.0 against v2.0.0 confirms that learned-model loading in `experiments/end_to_end/score_end_to_end.py`, the integration configuration, and evaluator implementations are byte-identical between tags. The changes in source are confined to size-refinement inference, partial-arrangement inference, geometry decoding, and constraint decoding. Added experiment files provide the scoring command and split utilities.

The final set's contract explicitly records `scored=true`, `scored_once="v1.0.0 and v2.0.0-rc1, not for model selection"` (`artifacts/v2_independent_eval/contract.json:30–33`). The release record identifies the final release's inference code as matching the scored candidate. This supports a fixed learned-base inference-bundle comparison, not an experiment showing that v2 neural training outperforms v1.

Frozen score rows do not themselves embed checkpoint digests or code commit IDs. Current release-record checkpoint hashes and unchanged loaders corroborate the recorded comparison, but the paper should not imply independently timestamped, per-row model provenance. The existing reproducibility language is appropriately cautious.

## Metric correctness

- Foreground count mismatch immediately returns failure (`src/knolling_model/evaluation.py:74–85`).
- Source identity requires center error <= 10 mm and oriented-envelope IoU >= 0.5 (`evaluation.py:92–106`).
- Predicted target poses are attached to true object geometry before scoring (`evaluation.py:116–125`).
- Held pose, axis, gap, and row tolerances match the manuscript (`arrangement/evaluation_v3.py:18–19`).
- The geometry validator uses oriented bounding rectangles and the 0.1 mm numerical allowance (`src/knolling_dataset/validator.py:30–63`).
- Binary task success requires good identities and no remaining error tags (`evaluation.py:130–134`).

Error tags are overlapping diagnostics, not causal effect sizes. Because count failure short-circuits all later checks, reductions in individual tags must not be interpreted as isolated ablations. The current paper explicitly avoids causal component attribution.

## Historical training and selection

The original selector summary agrees with the manuscript: 1,792 training families, 5,376 eligible full pairs, 5,085 usable, 249 association and 42 count exclusions, 1,272 updates over eight epochs, with epoch 1 selected on the 1,200-pair full validation pool. Its 222→235→233 trajectory and row/slot accuracies are correctly reported.

The broader joint pilot is recorded as FAIL / collapse with update 0 retained, and the 200-update diagnostic falls from 2/38 to 0/38 development and 24/152 to 1/152 validation. The manuscript correctly treats this as one failed configuration and does not claim general impossibility of end-to-end learning.

Historical `holdout` labels are hazardous: `experiments/end_to_end/dev_splits.py:17–30` defines a 640-example slice of training families, whereas the final independent set has 128 new families and five commands each. These are different datasets with the same record count. Existing appendix language keeps them separate. The earlier 260-scene diagnostic also overlapped historical training; it must remain excluded from generalization claims.

## Small editorial issue

At review start, Method contained an unmatched quote in “Vertical rotates objects; rows still run left to right.” Use “Vertical” in matched quotes, or simply “Vertical alignment rotates objects; rows still run left to right.” No experiment-result edits are needed.

## Recommended emphasis

The strongest supported experimental contribution is a paired, fixed-base increase in complete-scene validity under strict geometry and preservation checks, with uncertainty estimated at the source-family level. The main failure analysis should retain the 71/100 spacing result and the distinction between internally predicted dimensions and true-geometry scoring. Neither strict synthetic pass rates nor the micrometre-scale borderline example establish robot precision, visual quality, or physical feasibility.
