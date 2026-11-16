# Independent implementation review — 15 November 2026

Scope: README, adopted loader, perception/language/row/geometry/partial source, dataset generator, restricted joint trainer, current manuscript and prior audit. Original code, data, and checkpoints were read only. No inference, training, notification, or model mutation was performed. Source checkout: `535f14f`.

## Overall assessment

The manuscript is substantially faithful to the implemented system. In particular, its distinction between the end-to-end research goal, the RGB-and-language inference interface, and the adopted staged hybrid implementation is correct. The reported v1→v2 comparison should remain framed as a bundle of inference refinements over one learned base. It cannot substantiate the marginal benefit of attention, a size head, geometric decoding, or joint training.

## Corrections and refinements worth making

1. **Fix the opening Method quotation.** `paper_submit/main.tex:65` has an unmatched opening quotation before “Vertical”. Use: “Vertical describes the objects' long axes; rows still run left to right.”
2. **Use separate notation for the goal vector and gap.** Appendix uses `g` both for the 13-dimensional goal and a scalar gap. Reserve `g` for the vector and use `δ` for the requested gap. The ideal spacing equation is illustrative, not the actual equality enforced by the decoder: the implementation uses rotated-envelope offsets and ±2 mm acceptance bands, preserves feasible coordinates, and performs optional tightening (`geometry_decoder.py:122–181`, `:333` onward).
3. **Do not conflate initial constrained search and repair ranking.** Initial candidates are ranked by row plus pointer log-probability (`constraint_decode.py:158–195`). Repair candidates use row probabilities and area order (`:320–365`); color splitting also uses row scores (`:368–411`). “At most eight checked candidates per search” is correct. “Only eight structures are considered” is incorrect because row partitions can be enumerated before truncation.
4. **Keep the per-object selection branch separate from the goal vector.** The current manuscript already does this correctly; preserve it in any new diagram. The 13 dimensions encode keep, axis, region, gap, grouping and sort only. Learned per-object selection additionally reads object features and language slots (`goal_encoder.py:28–40`).
5. **If discussing fallback orientation, make the complete fallback chain explicit only in the Appendix.** PCA is preferred; a separately trained direction head is used if available, then the base detector angle if that checkpoint is absent (`size_refinement.py:207–250`). The current text's “learned fallback” is sufficient for the main text.
6. **Qualify size-order enforcement by route if elaborating.** The full route explicitly applies `apply_area_order` before geometry (`size_refinement.py:320–336`). The partial route does not apply that generic operation in its normal path; row repair uses area order if invoked (`partial_arrangement.py:182–224`, `constraint_decode.py:359`). Saying every output is deterministically sorted by area would overstate the code.

No material new unsupported result claim was identified in the examined TeX. Earlier audit file line references are occasionally stale: current `load_bundle` starts at line 52 and `load_adopted` at 69, for example.

## Concise mathematical explanation recommended for Method

The useful research idea is a factorization of discrete layout structure from metric coordinates, with selective preservation. It deserves one short equation in the main text rather than only module dimensions in the Appendix.

Let `o_i∈R^12` be detected object features, `g∈R^13` the corrected global goal, and `s_i` predicted dimensions. For full arrangement:

`h_i = φ(o_i),    h̃ = h + Attn(LN(h + W_g g + W_s s))`

`(r,π) = Select(h̃,g),    P^0 = Pose(h,g,r,π),    P̂ = Decode(P^0,r,π,s,g)`.

Explain that `r` is row membership and `π` within-row order; `Select` includes bounded candidate selection, while the later repair passes may revise structure. `Decode` uses predicted rotated rectangles and can retain an infeasible original when search is exhausted. This is a high-level factorization, not a claim of a globally optimized constrained objective.

Important accurate point: the contextual features `h̃` feed row/pointer heads only. Pose heads read original `h`; the selected discrete structure is how context affects coordinates. Source: `set_context.py:43–58`, `row_layout.py:250–274`.

An optional one-line geometric intuition is:

`x_(j+1) − x_j ≈ (w_j + w_(j+1))/2 + δ`,

where `w` is rotated horizontal extent and `δ` the instructed gap. State that the actual decoder enforces a tolerance band. This connects the size-refinement module directly to the dominant spacing failure mechanism without claiming measured causality.

## Narrative improvements backed directly by code

- **Describe paired instructions as controlled same-image contrasts.** Five commands share each source image; they contrast axis, referent selection, and one of gap, size order, grouping, or region. This gives the corpus a stronger explanation than its size alone. Generation records explicit contrast pairs and validates them (`expanded_data.py:44–71`). Do not imply all possible combinations were exhaustively covered.
- **Explain why scene-level validity is the demanding objective.** Local row/slot accuracy can rise while the last gap or identity failure still causes whole-scene failure. This is a coherent connection between the architecture, selector learning curve, and strict score; it is more informative than a long sequence of training-history details in the main text.
- **Position v2's contribution modestly but clearly.** An object-centric arrangement system and an auditable demonstration that sentence consistency, identity filtering, and feasible structural repair can improve complete-task reliability under a calibrated synthetic contract. Do not call the project a new foundation model, universal rearrangement policy, or a new end-to-end training method.
- **Make geometry uncertainty the focused remaining research question.** Decoder feasibility uses predicted size/pose while the evaluator uses true envelopes. This explains why geometric postprocessing is useful but cannot guarantee valid outputs; it supports a future dimension-aware or uncertainty-aware decoder, without inventing an experiment.

## Verified architectural details

| Claim | Current source evidence |
|---|---|
| RT-DETR-R18, 300 width-256 queries, 3 decoder layers, 640×640 internal image | `rtdetr.py:31–43`; `arrangement/perception.py:68–93` |
| Native input remains 576×384 RGB | `arrangement/perception.py:68–75` |
| Predicted-box 16×16 crop adapter, 288→128→10 head | `architectures.py:58–95` |
| Additional 8×8 crop size residual, 272→64→2 | `size_refinement.py:22–53` |
| 12D object vector with π-periodic orientation | `arrangement/contracts.py:48–53` |
| Text-only 13D goal; separate referent grounding | `arrangement/goal_encoder.py:28–40` |
| One 128D, four-head set-attention block | `set_context.py:11–40` |
| Context only changes structure; pose reads original features | `set_context.py:43–58`; `row_layout.py:250–274` |
| Neural row origin and positive local X increments | `row_layout.py:76–103`, `:179–203` |
| Row/origin dimensions 417→128→2; pointer 269→128→128 | `row_layout.py:23–40` |
| At most 8 checked initial candidates; at most 8 returned repair candidates | `constraint_decode.py:158–195`, `:320–365` |
| Partial placement translates a moving block around held objects | `partial_arrangement.py:129–158`, `:182–224` |
| Inference consumes observed image/text, no target labels | `size_refinement.py:298–340` |
| Model/data split created before rendering | `expanded_data.py:32–35`, `:38–74`, `:93–109` |

One subtle source detail is harmless for the adopted diagram: `NeuralRowPlacement` defines `row_anchor` and `row_gap` extensions, but the selected selector checkpoint has identically zero row-anchor weights/bias and zero final row-gap weights/bias. I checked this read-only by decoding tensor storage through a restricted unpickler (no torch/model execution). Thus those corrections are inactive in the adopted weights; depicting the operative row-origin path is appropriate. The joint checkpoint contains no such extension keys and the loader supplies zero initialization (`row_layout.py:48–62`).

## Exact end-to-end boundary

Recommended language:

> Our goal is end-to-end instruction-conditioned arrangement. The evaluated system realizes the RGB-and-language-to-pose interface through staged learning, discrete structure selection, and explicit geometric decoding; successful joint optimization of the complete pipeline is not established.

Evidence:

- The adopted entry point is `@torch.no_grad()` and reads image/text only (`size_refinement.py:298–340`). This is a deployment property, not itself evidence about past training.
- The restricted joint stage computes the size loss, then explicitly sets `replaced['size'] = size.detach()` (`joint_stage.py:159–167`). Perception/text extraction and goal/relation computation are under `torch.no_grad()` (`:123–125`, `:163–165`).
- Pose error therefore cannot update the size residual or perception. The angular penalty is in the scalar loss, but the angular head remains frozen; only row-coordinate parameters and the separately supervised size residual update.
- Discrete count filtering, row/order decisions, template correction, silhouette PCA, and bounded geometry/translation search are material in the final inference computation. They should appear as such in a diagram, not be styled as a single trainable network.
- The rejected broad joint pilot is valid negative development evidence. It cannot be treated as an ablation of final v2 or evidence that end-to-end training is impossible.

## Files changed by this review

Only this report: `paper_submit/audit/implementation_review_20261115.md`.
