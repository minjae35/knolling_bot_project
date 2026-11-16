# Final figure and caption consistency review

Reviewed the canonical `paper_submit/knolling_bot_v2_paper.pdf` with SHA-256 `683496ee4de3f6f74a0ed9d8c7f18ec9564597ba5bc3ad21a3ed429fdf916f04` (12 pages). This review inspected the six redesigned figure exports, their actual PDF pages 1, 2, 4, 5, 9, and 11, current root LaTeX captions and references, generating scripts, and frozen scientific evidence. It made no TeX, figure, model, dataset, or checkpoint edits and performed no inference.

## Verdict

**PASS: no material scientific or caption inconsistency identified.** All six changed figures agree with their captions and retained evidence. Figure 6 remains protected byte-for-byte. The illustrated validation records remain explicitly separate from the sealed quantitative evaluation.

## Figure-by-figure checks

### Figure 1 — language-guided arrangement (page 1)

- Both Blender panels contain six objects from `expanded_e2ee7a75ae3a34826b36d71f_0`.
- The preserved record reports PASS, horizontal long axes, ascending bounding-rectangle area, full workspace, no color-grouping requirement, and requested 30 mm horizontal/vertical gaps. Caption and panel status match these facts.
- The caption correctly identifies the scenes as a saved validation prediction, preserves the distinction between original model RGB and presentation renders, and does not imply robot execution.
- Verified the source-record hash and both PNG/Blender scene hashes against `teaser6_provenance.json`; all match. The rendering audit reports maximum transform/vertex discrepancy below `1e-7`.

### Figure 2 — architecture (page 2)

- The actual original RGB inset feeds the perception arrow; the surrounding Blender views are identified as presentation views at stored poses.
- Object/row glyphs are expressly schematic. Their arrangement is not claimed to be an activation visualization or an additional prediction.
- The global goal `g(t)` is text-only; the distinct `m(O,t)` selection branch also reads objects. The language arrow routes to attention, not to the relation MLP.
- Contextual features feed row and pointer structure. The separate lower route feeds original relation features, goal, and sizes into pose heads. This agrees with actual model routing and the caption.
- Learned modules and explicit rule processing have distinct colors. Held/no-op routing is identified as estimated source position plus image/PCA angle, not copied ground truth.
- Source-image hashes in `architecture_style_revision.json` match current files. The surrounding text retains staged optimization and lack of guaranteed geometric feasibility.

### Figure 3 — paired outcomes and failures (page 4)

- Paired bar counts are 489 retained successes, 51 gained, 100 persistent failures, and zero regressions; they sum to 640 evaluated image–instruction pairs.
- The +8.0 percentage-point headline and +4.8 to +11.6 interval correctly round the independently reproduced 7.96875-point estimate and [4.84375, 11.5625] family-bootstrap interval.
- Object-count curves use the verified 90/95-pair denominators and correctly label the eight-object values 52.2% and 72.2%. The caption says success generally decreases, without claiming monotonicity or a statistically established interaction.
- The final-failure partition is mutually exclusive: horizontal gap only 29, both gaps 20, vertical gap only 22, and neither gap 29. Hence 71/100 failures include a spacing violation. The caption explicitly permits other error tags to coexist.
- The figure presents the fixed-base release comparison, not additional training or component ablations.

### Figure 4 — correspondence and motion ledger (page 5)

- Both panels show the eight-object saved validation record `expanded_717287879ce16878c5558650_4`; the recorded PASS/right-region/color-row status matches the panel line.
- The new caption and Appendix D correctly describe projected object IDs on paired Blender views. They no longer describe the removed world-plane displacement map or center-to-center arrows.
- Independently recalculated all eight translations and signed rotations directly from the saved source and predicted poses. They exactly match `motion_annotations.json` before rounding. Displayed translations in ID order are 14.0, 203.9, 135.6, 282.7, 197.8, 224.2, 206.3, and 211.6 mm; rotations are +3.9, +11.4, −42.0, −69.9, −84.5, −69.9, +17.5, and +72.9 degrees.
- Axis glyphs are correctly labeled as world-plane directions, positive counterclockwise and 180-degree periodic. Photo leader lines identify objects; neither text nor caption interprets them as robot trajectories.
- All five source hashes in the motion audit match. Its saved-camera basis error is approximately `1.94e-8`.

### Figure 5 — historical selector training (page 9)

- Both curves come from the preserved eight-epoch selector result on 1,200 full-arrangement validation pairs. They are explicitly labeled as historical validation before final inference refinements.
- The success counts by epoch are 235, 235, 233, 228, 229, 224, 232, and 233. Epoch 1 is the recorded selected epoch; its highlighted marker and shading are correctly described.
- The dashed baseline is the genuine recorded 222/1200 = 18.5%, not a fabricated reference line. The caption and nearby prose agree.
- Final row/slot accuracies round to 84.9%/74.7%; the full-precision prose gives 84.88%/74.68%. Their increase is correctly distinguished from plateauing complete-scene validity.

### Figure 7 — failure margins (page 11)

- Caption correctly describes the current two-column Blender closeups and numerical rulers below them.
- Geometry remains derived from preserved poses and true oriented rectangular envelopes: row gap 17.6573726958 mm, 0.6573726958 mm beyond the 17 mm upper limit; minimum clearance 9.8961028328 mm, 3.8971671752 µm below the effective 9.9 mm minimum.
- The full interval 13–17 mm correctly represents a 15 mm request with ±2 mm scoring tolerance. The clearance rule explicitly subtracts the 0.1 mm numerical allowance from the nominal 10 mm minimum.
- Dashed guides refer to evaluated envelopes, not chamfer/capsule silhouettes. The positive clearance does not imply an envelope collision; the figure correctly describes a borderline clearance failure.
- The separate ruler scales are labeled. The approximately 0.012-render-pixel deficit is a scale explanation, not a claimed physical positioning precision.
- Source-image/evidence hashes, output hashes, and generator hash in `failure_style_revision.json` match current files.

## Protected Figure 6

Ran `paper_submit/.venv/bin/python paper_submit/scripts/preserve_figure6.py` read-only. Result: PASS; all 11 protected files and the complete LaTeX figure block match `audit/figure6_frozen.json`. No regeneration or modification was performed.

## PDF integration

The redesigned figures and captions are present on their expected pages. Visual inspection found no clipped scientific label, missing panel, wrong figure reference, or apparent caption/figure mismatch on the six reviewed pages. Page count and PDF digest match the requested review target. The manuscript continues to distinguish synthetic target-layout validity from robot execution and an end-to-end learning objective from the adopted staged hybrid system.
