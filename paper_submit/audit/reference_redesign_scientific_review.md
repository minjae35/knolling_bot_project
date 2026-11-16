# Scientific review of the reference-based figure revision

**Status: PASS.** No remaining material scientific misstatement, caption mismatch, or misleading geometric representation was identified in the reviewed scope. Two clarifications requested during review—“pointwise” intervals and “largest observed gain”—are present in the final manuscript.

Canonical PDF: `paper_submit/knolling_bot_v2_paper.pdf`  
SHA-256: `c430318c11b07dcd0283e93af6e0ef6d03535dd97ae6f402f787efa0582ae4be`  
Length: 12 pages; main text remains pages 1–5.

## Review scope

Read the current figure captions, surrounding main text, and Appendices C–E. Directly inspected rendered pages 4, 5, 9, 10, and 11; after final editorial refinement, reinspected pages 4 and 5 in `build/visual_review/reference_redesign_v2/`. Checked the preserved raw records, analysis reports, and figure construction code. Independently recomputed the new footprint/envelope distances and Figure 4 row/region/ordering quantities using a separate scalar rotation implementation and Shapely, rather than the figure helper functions.

This reviewer implemented Figure 3 earlier in the revision; its plotting review is therefore a self-review, not an independent authorship audit. The geometric recomputation is independent of the Figure 4 and Figure 7 generators. This final review changed only this report and performed no model inference, training, or scientific-data edits.

## Figure 3: paired gains and retained bottlenecks

The panel now asks a coherent empirical question: which instruction modes gain, and which recorded error categories persist? All four mode estimates use matched release outcomes, not independent samples. Resampling retains all five related commands in each selected family. Mode-specific denominators are recomputed within each family resample, correctly handling the 127 partial and 129 no-op records.

Verified estimates and 95% percentile intervals, in percentage points:

| Mode | Gain | Interval | Records |
|---|---:|---:|---:|
| Full | 6.250000 | 2.864583–10.156250 | 384 |
| Partial | 15.748031 | 9.523810–22.047244 | 127 |
| No-op | 5.426357 | 1.562500–10.000000 | 129 |
| Overall | 7.968750 | 4.843750–11.562500 | 640 |

The overall interval exactly matches the existing aggregate audit. Seed 20261112, 10,000 family resamples, and sorted indices 250/9749 are documented. The final caption explicitly says **pointwise**; the text says **largest observed gain**. It makes no simultaneous-coverage claim or hypothesis-test claim that partial improvement exceeds full/no-op improvement. Uncertainty remains conditional on the two frozen releases and does not stand in for training-seed or domain-shift uncertainty.

The nine error comparisons match the raw records: gap union 77→71, count 45→5, region 17→4, size order 13→11, preservation 9→4, clearance 9→7, mixed-color row 2→0, collision 1→0, and boundary 1→0. “Any gap” is a union, not a sum. Error categories may overlap. All 45 v1 count-failure records and all 5 v2 count-failure records emit only the count tag; the caption correctly warns that count failure skips later checks. The plot consequently describes emitted tags, not uncensored latent violations or individual rules’ causal effects.

At paper width the zero-valued endpoints remain visible with the final axis padding. The before/after connectors are distinct from the left panel’s confidence intervals. Labels, point estimates, and interval caps remain legible. Page 4’s analysis and last lines fit without clipping.

## Figure 4(c): instruction constraints at the saved layout

The complete workspace overview, shaded right half, and connected enlarged plan view explain spatial constraints more directly than the earlier pose-change scatter. The caption identifies the view as an audit using recorded poses and true geometry; it does not claim neural activations, a new prediction, or measured latent row assignments.

Independent reconstruction confirms:

- Rows, top to bottom: `(o03, o08)`, `(o01, o06)`, `(o07, o05)`, `(o04, o02)`.
- Vertical envelope-band gaps: 30.5918478774, 15.1589829323, and 31.6684855654 mm, within requested 30/15/30 mm intervals at ±2 mm.
- Horizontal gaps: 16.2652317973, 15.7950606854, 14.2433962540, and 16.1067896409 mm, all within the 15 ±2 mm requirement.
- Envelope x bounds: 0.3210271222–0.4905441647 m, within the right region [0.3, 0.6] m.
- Bounding-rectangle area is descending within each color in the displayed traversal order; this is the area definition stated by the source instruction.

Both overview and detail use preserved target poses and uniform geometric scales. Their connecting crop lines map the same region. Stable IDs link the plan detail to the paired Blender views. The annotated bottom-row horizontal gap is a representative measured pair, not a fabricated global average. The caption and Appendix D correctly describe the selected validation record and refrain from claiming a collision-free execution path. Page 5’s conclusion remains fully visible.

## Figure 7: footprint distance versus scored envelope clearance

The revised figure makes a scientifically useful distinction: original shape footprints and scored bounding rectangles can have different separations at the same poses. The detailed geometry is linked to actual Blender context rather than presented as an unrelated schematic.

Independent computation over all 28 object pairs in the partial scene confirms that **o06/o08 is the global minimum pair under both representations**:

- Original footprint distance: **12.8198497339521 mm**.
- Oriented rectangular-envelope distance: **9.89610283282485 mm**.
- Effective required clearance: 9.9 mm; shortfall approximately **3.897167 μm**.

The 12.820 mm claim is thus supported by direct geometric reaggregation of stored predictions, not a changed evaluator or new model evaluation. Figure and caption consistently state that the tiny threshold deficit is below render resolution. Neither the text nor the figure treats positive envelope separation as a collision, claims real-world micrometer precision, or establishes physical feasibility. The text also avoids estimating the frequency of similar failures across the holdout.

For the full-layout example, the 17.6573726958 mm row gap exceeds 17 mm by 0.6573726958 mm. The allowed 13–17 mm edge strip is drawn in the same world coordinates as the objects, at its actual width. Dashed envelopes, solid footprint outlines, and the allowed band have a common representation key. Page 11 is dense but its last paragraph, annotations, and figure caption remain inside the text area.

## Remaining pages and protected content

Page 9 accurately documents the new mode-specific denominator while retaining the historical selector diagnostic’s separation from final evaluation. Its final error-table row remains fully visible. Page 10 distinguishes derived geometry from latent structure and retains pose-change quantities only as archived measurements.

The Figure 6 preservation check passes: all 11 protected export/render/scene hashes and its complete LaTeX figure block are unchanged. No scientific evidence or visual styling change was made to Figure 6.

## Review conclusion and limits

The revised claims are supported by stored results and explicit geometry, with uncertainty and evaluation limitations stated. No remaining required scientific correction was found. This review does not predict a grade, venue acceptance, or universal visual preference. The manuscript remains a synthetic, fixed-release study without physical execution or isolated causal ablations.
