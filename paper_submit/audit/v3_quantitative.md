# Quantitative redesign — 2026-11-15

The main results figure now answers three different questions in a 396 × 200 pt composition. Entry point: `scripts/quantitative_figure.py::generate()`.

- **Paired outcomes:** 489 retained successes, 51 gains, no regressions, 100 persistent failures among 640 pairs. Segment widths are proportional to counts. The gain is 7.96875000 percentage points, with the existing paired family-bootstrap interval [4.84375, 11.5625] pp displayed to one decimal. This is not an interval for independent trials or training seeds.
- **Scene complexity:** the saved success curves for 2–8 objects remain, using the full 0–100% scale. Bins contain 90 or 95 pairs; no uncomputed uncertainty bands are invented.
- **Spacing:** the 100 v2 failures are partitioned into 29 horizontal-only, 20 both-gap, 22 vertical-only, and 29 neither-gap scenes. Thus 71 have one or more spacing violations. Unlike a sum of raw error tags, these four bars are mutually exclusive. A scene in a spacing category may additionally violate another condition; “neither gap” does not mean absence of other failures.

The paired counts and partition are independently rederived from frozen JSONL records on every generation. Source hashes are checked against `audit/verified_results.json`. The original data, model, and predictions are not modified. All labels are at least 7 pt; the composition is exported to vector PDF/SVG and a 300 dpi PNG. Text extents are checked against the canvas before export.

Suggested caption: **Paired release gains and remaining difficulty.** (a) All 489 v1 successes remain valid; v2 gains 51 scenes and loses none. The interval resamples the 128 paired families. (b) Success generally decreases with object count (90–95 pairs per bin). (c) Gap categories partition the 100 remaining failures: horizontal-only, both, vertical-only, or neither. Other error types may coexist with any gap category.

The exact mode counts remain in the adjacent manuscript table. This figure intentionally does not repeat that complete table. All nonzero raw error tags remain documented in the manuscript and appendix.

Input SHA-256:

- `v1_holdout.jsonl`: `89b3ee1cbc2433b948bdb0b85ac7a2d407b3001a90018297d94dd25266218aaa`
- `v2_holdout.jsonl`: `209df3c1b12cb62e82ce7daff066cc64ced99e26c9960e2b5e1070255ac5de82`
