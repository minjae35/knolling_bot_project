# Figure-set diagnosis and design brief — 2026-11-15

Read-only inspection of the current working manuscript, its figure sources, the README demonstration, and primary research papers. This note records a diagnosis; it does not claim that the recommended redesign has already been implemented.

## Concrete problems in the current artifact

The current `knolling_bot_v2_paper.pdf` has **13 pages**, with **six main-text pages** and references on page 7. The 340 pt opening figure pushes most of the introduction onto a mostly empty page 2, followed by a forced page break. This is a measurable layout regression, not a subjective preference. The original bold two-line title, stacked name/department/university, and absence of running headers are intact and should remain.

1. **The architecture has again become generic cards, with incomplete connectivity.** The headings are misaligned with content: “Language” sits over the object matrix. The object set has no drawn route into the Relation MLP, the goal does not connect to set context, row/order does not feed pose heads, and preservation does not merge with the output. The curved original-feature branch crosses the attention module and its text. Detached rule badges below the render convey a list, not the actual inference order. Some labels are only 5.8–6.6 pt at publication size. The network is simplified beyond the level required to explain the actual engineering.
2. **Motion information is split and repeated.** Figure 1 shows source/target renders plus a very tall planar diagram, while Figure 4 repeats the same source/target renders and adds a ledger. The figure with vectors does not label source footprints consistently, has no visible per-object rotation information, and draws opaque targets over source outlines. The figure with rotations no longer has movement arrows. Readers must reconstruct information across pages rather than read one coherent motion figure.
3. **The physical appearance is distorted in one compositor.** `motion_figure.py` places native 1920×1280 images into 192×108 rectangles with `aspect='auto'`, which compresses the vertical dimension. Correct paper artwork must preserve the native 3:2 aspect, or use a documented common crop while preserving isotropic scale. Architecture output thumbnails are also placed at 96×52 rather than 3:2.
4. **More pixels did not produce more effective evidence.** The publication renders add a broad tray border, large flat background, and weak directional shading. At paper scale the arranged objects occupy only a small part of each panel. The README demo has stronger color/shadow separation and a more direct overhead view. Any rerender should improve object/background separation and useful occupancy, with a shared camera/framing/material/light setup for both poses. Do not alter geometry or target positions.
5. **Command comparisons again label failure without locating it.** The latest `publication_figures.py` removed the prior in-image callouts. A compact status under the panel is useful but cannot show which row gap or object pair violates a constraint. The separate failure figure is substantially stronger: it should remain the diagnostic source of truth and be visually linked by corresponding IDs/markers.
6. **Quantitative plots are competent but the main effect is not visually foregrounded.** The current figure emphasizes object-count and overlapping error tags, whereas the central evidence is a paired release improvement with a fixed learned base: 489 preserved successes, 51 gains, zero regressions, 100 persistent failures. A carefully integrated paired-outcome graphic can explain this more directly than decorative bars. Keep uncertainty family-aware and do not invent per-bin intervals.

## Primary figures actually inspected

PDFs were retrieved from their primary arXiv pages and rendered locally for image inspection. Reference renders were written only to `/tmp/knolling-v3-design/`.

- [DETR](https://arxiv.org/pdf/2005.12872), Figure 2 on PDF page 7: backbone features, positional encoding, object queries, token routing, and parallel prediction heads are visible as different graphical objects. Module groups have clear boundaries, but the figure is not merely a row of named boxes. For Knolling, use explicit feature dimensions and separate row/pointer/pose heads; preserve the original-feature bypass. The goal is visible neural computation, not copying DETR's architecture.
- [StructFormer](https://arxiv.org/pdf/2110.10189), Figure 2 on PDF page 3: actual language and object tokens align with latent/position/type rows, and selection is distinct from autoregressive pose generation. The use of actual example tokens grounds technical structure. For Knolling, show the object-conditioned goal and selected/held branches explicitly. Do not inherit its 6-DoF or execution claims.
- [DINOv2](https://arxiv.org/pdf/2304.07193), Figure 3 on PDF page 4 and Figure 1 on page 2: stage separators and actual image thumbnails make an overview understandable, while registered image/result pairs make qualitative correspondence immediate. This is useful for the overall pipeline and qualitative layout; it is insufficient by itself to depict internal neural structure.
- [Segment Anything](https://arxiv.org/pdf/2304.02643), Figure 1 on PDF page 1 and Figure 2 on page 3: task, model, and data receive different visual encodings, and qualitative examples devote most space to results rather than labels. The lesson is purposeful variety. A repeated three-card template across every figure would miss that lesson.

These are visual-design references, not additional baselines or evidence for this project's results. No artwork should be copied.

## Recommended figure-specific design

### First-page task example

Keep a compact matched **3D before/after pair**, approximately 160–180 pt high, with a short actual command and small stable IDs if useful. Let object appearance carry this figure. Remove the large planar movement panel from the opening page. A task teaser is not the place to explain all eight rotations. This should restore a complete first-page introduction and a five-page main body without shrinking text.

### Architecture

Use two coordinated layers with distinct purposes. The overview shows RGB and English, their actual representation outputs, selected/held split, learned arrangement, deterministic geometric processing, and final target-pose merge. The detail shows actual neural substructure: 12→128→128 relation MLP, goal/size projections into one four-head set-context block, row logits and autoregressive pointer order, and pose heads reading original relation features plus required structure/goal/size inputs. Show named dimensions and branching where informative, not arbitrary token stacks. Route arrows around labels and keep every relevant connection explicit. A small legend distinguishes **learned parameters** from **deterministic inference**; training-stage freeze boundaries belong in a separate strip or appendix, not an ambiguous blanket “trainable” mark. Minimum text should be around 7.5 pt at final paper size.

### Dedicated motion figure

Prefer **one large, registered source/target footprint overlay** plus a compact eight-object translation/rotation ledger, rather than repeating the same two renders. The earlier teaser supplies the photorealistic appearance. Use dashed/light source footprints and solid outlined targets, not opaque targets that hide source states. Label IDs at both source and target with carefully placed leaders; preserve actual positions and angles. Include long-axis marks and one unobtrusive illustrative angular arc if it helps; keep all exact signed rotations in the ledger. Straight arrows connect centers and must be labeled displacement vectors, not robot trajectories. The attached screenshot and README center panel are appropriate references for this figure only. At 396 pt width, copying its three wide panels literally would make labels too small.

If overlap cannot be resolved without moving data, split the motion overlay by object color or use small per-object pose glyphs, while retaining a full-scene map for absolute movement. Avoid fake curved travel paths and avoid shifting targets for cleaner composition.

### Qualitative command comparison

Use a shared initial scene and the three actual command outcomes with constant image scale and camera. Focus this figure on semantic selection/preservation: use object IDs or a subtle selected/held legend, not motion arrows everywhere. Add small numbered diagnostic markers that link to the failure closeups. Clearly distinguish `borderline clearance failure` from a visible collision. Different instructions should remain readable and recorded, not invented paraphrase tests.

### Failure analysis

The existing crop-plus-threshold-ruler concept is sound and substantially more effective than a FAIL label. Keep the evaluated rectangular envelopes distinct from mesh silhouettes. Retain the explicit 17.657 mm row gap, 0.657 mm excess, and 9.8961 mm clearance with 3.9 µm deficit. Use a marker showing where the crop belongs in the whole arrangement. The micrometer-scale case must remain a borderline evaluator result, never an implication of physical positioning precision. Avoid devoting excessive main-text area to a tiny numerical edge case while omitting broader failure categories.

### Quantitative and training results

Use a paired-outcome summary or a directly labeled before/after comparison with the overall +8.0 pp family-bootstrap interval; the same learned base and paired nature are the important result. Preserve mode counts in a compact table if they add exact denominators. Scene-complexity and error-tag diagnostics can remain a separate focused panel or appendix. Keep the selector learning curve as a two-axis small multiple: structure improves while complete-task success plateaus. Do not visually equate its historical 1,200-scene metric with the final 640-pair benchmark.

## Recommended acceptance gate for the first core figure

1. Select one core figure and complete its scientific content before restyling other figures.
2. Insert it into the actual LaTeX page at final width; inspect both a full page and a 100%-scale crop.
3. Require no aspect distortion, no clipped or crossing labels, no unconnected mandatory branch, no text below the agreed minimum, and no hidden source–target identity.
4. Verify the figure's values/transforms against the saved records, then lock its design vocabulary where appropriate.
5. Use that successful figure as a quality threshold, not a template to duplicate across all figure purposes.
6. Re-render every final page and check the five-page main-text boundary, caption lengths, sparse forced pages, and duplicated renders.
