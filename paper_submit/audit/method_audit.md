# Method and training evidence audit

Audit scope: read-only inspection of current source and saved JSON/log artifacts, 2026-11-15. No training or model inference was run; the sealed holdout was not rescored. Paths are relative to the repository root. This memo is evidence for manuscript drafting, not a claim that every historic run can now be reproduced.

## Executive conclusions

1. The original research goal can honestly be described as **end-to-end language-conditioned knolling**. The implemented result is **end-to-end inference from RGB and text to target poses**, using a **staged, hybrid learned-and-geometric system**. It is not a fully end-to-end-trained neural policy and it does not execute a robot.
2. The actual adopted method is richer than the README diagram: RT-DETR with an appearance ROI adapter; separate object-size residual; MiniLM plus goal and referent heads; inherited relation features; one object-set attention layer; neural rows/order/initial coordinates; constrained search, area ordering and geometry corrections; partial/no-op routing; RGB silhouette PCA for angles of held objects.
3. The 13-dimensional goal does **not** itself contain per-object selection probabilities. Those are a separate head. Correct decomposition: keep 1 + axis 2 + region 3 + gap 3 + grouping 2 + size order 2 = 13.
4. The joint stage explicitly blocks pose gradients into size/perception. A separate broader `end_to_end_joint` experiment failed after 200 updates and was not adopted. That is useful negative evidence supporting the staged design, not proof that end-to-end training is impossible.
5. The release is a pipeline comparison. Do not attribute the entire v1-to-v2 holdout gain to a single learned module, or describe it as a controlled component ablation.

## Actual inference chain and exact implementation references

- `experiments/end_to_end/score_end_to_end.py:46` (`load_bundle`) loads integrated perception/arrangement, relation checkpoint, row model and size residual. `:64` (`load_adopted`) loads the **joint-stage** row/size checkpoint, overlays selector placement weights and adds `SetArrangement` context weights. It does not run the original relation pose head.
- `src/knolling_model/experiments/integration.py:88` (`load_integrated`) checks checkpoint SHA-256 and overlays the perception and arrangement bases. Base paths and updates are in `configs/structural_integration.json` (perception update 2400; arrangement update 6000).
- `src/knolling_model/experiments/size_refinement.py:299` (`poses_from_detection`) is decorated `@torch.no_grad()`. Inputs are system objects plus images and text. It reads **no true source pose, true size, true object count, teacher row, teacher order or target pose**. Source query IDs are frame-local detector identities; evaluator-only ground-truth association is external.
- Input images are 576 x 384 RGB in [0,1], but RT-DETR internally resizes them to **640 x 640**; see `src/knolling_model/rtdetr.py:48` and `src/knolling_model/arrangement/perception.py:72`. The detector has 300 queries, 256-dimensional query tokens and three decoder layers. Do not claim detection operates internally at native 576 x 384.
- `src/knolling_model/experiments/architectures.py:54` (`AppearancePerception`) adds a predicted-box **16 x 16 RGB crop** adapter. CNN channels 3→16→32, pooled feature 32, query+feature 288→128→10 residual head. Residuals adjust size2, angle1, color3, shape3, foreground1. Crop geometry comes from predicted boxes and is detached. The adopted detector is therefore not an unchanged off-the-shelf RT-DETR.
- `size_refinement.py:22` adds a separate **8 x 8 crop** size residual: 3→16 convolution + GELU + average pool; concatenated with 256-dimensional detector token; MLP 272→64→2. The output is `(base_size + residual).clamp_min(1e-4)` metres. Zero-initialized final readout initially preserves the base predictor.
- `size_refinement.py:257` suppresses the lower-scoring detection within 10 mm of a retained center, among foreground scores ≥0.5. This is a postprocessing rule; do not repeat RT-DETR's generic “NMS-free” claim for the entire final system. Inference rejects counts outside 1..8; generated/evaluated data cover 2..8.
- `src/knolling_model/arrangement/contracts.py:47` constructs the **12-dimensional** object vector: `[x/.6, y/.4, major/.6, minor/.4, cos(2θ), sin(2θ), color_probs(3), shape_probs(3)]`.
- `size_refinement.py:103` tokenizes the English sentence and obtains MiniLM last hidden states (384 dimensional). `src/knolling_model/arrangement/goal_encoder.py:10` projects 384→128, uses seven learned language queries, 4-head attention, residual normalization and FFN, and grounds object referents through `JointBlock`. The vector is `[p_keep, (1-p_keep)p_axis, (1-p_keep)p_region, (1-p_keep)p_gap, (1-p_keep)p_group, (1-p_keep)p_sort]`; an independent sigmoid head supplies per-object selected probabilities. Five attribute vocabularies appear in `goal_labels.py:7`.
- `geometry_decoder.py:185` uses exact template phrases to pin region and gap **only when the learned argmax disagrees**, preserving the field's mass. Thus language processing is partly learned and partly template-based. Partial placement additionally pins the requested axis (`partial_arrangement.py:189`, `geometry_decoder.py:216`). Full commands are explicitly recognized by `startswith('Arrange all objects')` to avoid wrong routing.
- `src/knolling_model/experiments/relation_pose.py:16` defines a 12→128→128 MLP (via instantiated `feature_dim`); the final pipeline uses only `relation._hidden(objects.features())`. The relation model's original autoregressive coordinate head is a training-history component, not the final pose predictor.
- `src/knolling_model/experiments/set_context.py:11` has **one 128-dimensional, 4-head object self-attention block**. Goal and size projections are added to tokens, with a null key to accommodate padding. Its residual output feeds the **row/order selector only** (`:43`); coordinate heads keep the original relation hidden states.
- `src/knolling_model/experiments/row_layout.py:22` defines an 8-class row head and within-row autoregressive pointer. Pose heads predict one row origin, positive horizontal steps conditioned on neighboring widths and requested gap, and π-periodic angles. Row-origin input is 417 dimensions: 3 × (128 hidden + 4 row statistics) + 13 goal + 8 row code. The per-row statistics are count, total width, max height, first-object width. `:238` distinguishes selector features from pose features. Teacher rows/order are optional training arguments; inference supplies `None, None, True`.
- `constraint_decode.py:158` ranks alternatives by summed row and pointer log probabilities and checks at most **8 candidates**. Feasibility uses predicted sizes, angles and command. `:236` enforces size order by predicted rectangle area. `:320` enumerates row partitions, ranks them with selector probabilities and returns at most 8 repair candidates. This repair can enumerate many partitions before limiting candidates, so do not say total search has only eight states.
- `geometry_decoder.py:122` solves coordinates using oriented-envelope offsets, requested gap bands, shared row y and region constraints. An acceptable pose is normally retained; horizontal and vertical tightening passes and a 1 mm region inset follow (`:333`). If a geometry proposal is infeasible, the original pose is retained with an explicit status. `constraint_decode.py:483` applies row repair only on infeasible layouts; `:414` can split mixed-color rows when requested and color confidence permits.
- `partial_arrangement.py:43` replaces the moving set with uniquely predicted objects of the unambiguous named color. `:182` arranges only movers. Held objects retain estimated source positions and angles. `:129` searches rigid translations on a **5 mm grid within ±120 mm** to keep movers in-region and ≥10 mm from held objects; this is a bounded obstacle-clearance heuristic, not motion planning.
- `size_refinement.py:157,232` estimates held/no-op object orientation from **RGB connected-component silhouette PCA**: chroma threshold 25/255, local ±80 mm window (80 mm radius, 160 mm span), at least 40 pixels. The major eigenvector gives π-periodic angle. When unavailable, it uses the separately trained direction head (or base physical angle if that checkpoint is absent). Centers remain detector centers. This explains why no-op can fail even though the system copies its estimated pose.

## Suitable concise method equations

Let `o_i` be the 12-dimensional detected object feature, `h_i = MLP(o_i)`, `g ∈ R^13` the goal, and `h'_i = h_i + Attention(LN(h_i + W_g g + W_s s_i))`. The selector predicts rows and within-row orders from `h'`, while initial pose heads receive `h`.

For adjacent axis-aligned target rectangles in a row, exact requested spacing has `x_{j+1}-x_j=(w_j+w_{j+1})/2+g_req`; between rows, `y_r-y_{r+1}=(H_r+H_{r+1})/2+g_r`, where `g_r` doubles between distinct color bands. These equations explain the decoder's geometric intent; implementation uses general rotated-envelope offsets and ±2 mm bands, and preserves feasible neural coordinates. Do not portray the simplified equation as a differentiable global training objective.

A truthful high-level map is `P_hat = Decode(Select(Context(Detect(I), Encode(t))), Detect(I), t)`, with separate keep/partial routing. State explicitly that inference has observed RGB/text only and all learned modules were optimized in stages.

## Training facts supported by saved evidence

| Stage | Evidence / configuration | What is justified |
|---|---|---|
| Perception and arrangement bases | `configs/structural_integration.json`; source `integration.py:88` | Independently frozen base checkpoints, updates 2400 and 6000. Avoid assuming all 20,000 samples were used. |
| Sequential/relation | `configs/neural_sequential_phase1.json`; `configs/phase1_relation_pose.json`; `scripts/run_phase1_pairwise_affine.py` | 791 train / 94 validation manifest; seed 1203. Relation objective combines teacher-forced order CE, absolute anchor Huber, neighbor displacement Huber (10 mm scaling), and angular error (`relation_pose.py:138`). |
| Size residual | `artifacts/experiments/size_refinement/result.json`; `experiments/size_refinement/run.py` | 3200 source families, 2560 fit/640 dev; seed 1203; batch 8; LR 1e-3; adopted update 200; all other modules frozen. Separate size supervision. |
| Restricted joint row/size | `artifacts/experiments/joint_stage/result.json`; `experiments/end_to_end/joint_stage.py` | Batch4, AdamW LR1e-5, weight_decay0, 30 updates, 120 attempted scene presentations, six excluded. Trainable size residual, row step, row-origin last layer; everything else frozen. Training families split 4/5 fit and 1/5 dev from official train only. |
| Set-context selector | `artifacts/experiments/selector_full/preflight.json`, `result.json`, `train.log`, `curve.jsonl` | 1792 eligible train families = 5376 full instructions; 5085 usable (249 alignment +42 count exclusions). 139,016 trainable parameters logged in smoke audit. 8 epochs/1272 updates, LR starts 1e-4; best epoch1. Row+pointer CE are recorded, with train_loss exactly their sum. 1200 full validation instructions used for checkpoint ranking. The original `train_selector_full.py` named in train.log is absent from current source tree, so exact retraining schedule cannot be fully re-executed from this tree alone. |
| Broad joint-training attempt | `artifacts/experiments/end_to_end_joint/result.json`, `history.jsonl` | Not adopted. Status FAIL, stop collapse, best update0. At update200: development success fell 2/38→0/38; validation success 24/152→1/152. This is one failed pilot with specific configuration, not an ablation of final v2 or evidence of fundamental impossibility. |

### Restricted joint loss and gradient separation

`joint_stage.py:117` extracts frozen visual/text features, matches detections to true **source** centers (≤10 mm), and excludes count/alignment failures. Ground truth is used for supervision only.

`L = L_size + L_pose(failed_reference_objects) + L_anchor(good_reference_objects)` (lines159–178,342).

- `L_size = SmoothL1(pred_size, true_size; beta=.001 m)`.
- Refined size is **detached** before packing layout objects (line161).
- `L_pose` is XY SmoothL1 beta .010m plus absolute periodic angular error; see `row_layout.py:288`.
- “Good” reference objects have position error <2mm and angular error <3°. Their new coordinates are anchored to frozen baseline via SmoothL1 beta .001m.
- Frozen detector, language, goal and relation evaluation are under `no_grad`. Row assignment, pointer and angle heads remain frozen. Gradient probes explicitly check isolation (joint_stage.py:250 onwards).

Therefore calling this stage “joint fine-tuning” requires the qualifier “restricted to size and row-coordinate heads with separated gradients.” The size head receives no pose gradient in this adopted stage.

## Development findings that may strengthen the paper, with correct scope

- Size residual development: the 152 full-scene diagnostic improves **24/152→42/152**; median major-size error **1.419→0.790 mm**, p95 **3.895→2.294 mm**, matched-object n=796. Source: `artifacts/experiments/size_refinement/result.json`. These are historic development diagnostics, not independent holdout performance.
- Restricted joint stage: same named 152-scene diagnostic **42/152→54/152**, vertical-gap median **2.015→1.588 mm**; source `artifacts/experiments/joint_stage/result.json`. Since each stage used evolving code and its own selection gates, label these as a recorded development sequence, not a fresh controlled factorial ablation.
- Selector: full validation **222/1200→235/1200**, mixed-color-row error counts **234→22**, row accuracy **68.52→75.86%**, slot accuracy **61.87→71.77%**. Best epoch1 despite training loss continuing to fall through epoch8 and validation row accuracy increasing to84.88%; whole-scene success at epoch8 is233/1200. This shows local assignment accuracy does not guarantee the strict scene-level objective. Source selector result/curve.
- Selector development subsets: Dev256 **79→88**, Dev512 **146→162**, but some scenes regressed (5 and10, respectively). Do not apply the final holdout “no regressions” claim to all intermediate experiments.
- Earlier broad joint pilot records overlap of a historical 260-scene “holdout” with older fitting data. It was not used for that pilot's selection. This does **not** describe the later independent640, but is a reason to omit that old holdout from claims of independent generalization.

## Reproducibility and scope limitations to disclose

- Checkpoints/datasets do exist locally in this workspace but are gitignored and not redistributed by the repository. Saying “not present” would be wrong; say “not distributed in the source repository.”
- Current loader transitively needs configs/current_models.json, recovery config, structural manifest, pinned HF snapshots, parent blob directories, selector selection/direction heads, relation, row and size weights. Release document lists hashes, but no download URLs.
- Frozen current source is not a verified full retraining recipe; missing historical selector trainer makes that limitation concrete.
- Simulated colors/shapes, calibrated top-down workspace, closed instruction templates, max8 objects; no real images, robot execution, grasping, trajectory planner or open-vocabulary claim.
- Goal was end-to-end; final achieved interface is end-to-end. Training is modular and deterministic postprocessing is material. Suggested sentence: “Our goal was end-to-end instruction-to-arrangement prediction; the resulting system realizes this interface with staged learning and constrained geometric decoding, rather than a single jointly optimized neural policy.”

## Verified v1.0.0 → v2.0.0 source delta (tag-to-tag)

Read-only command: `git diff v1.0.0 v2.0.0 -- src scripts experiments configs`.

The only changed inference modules are `constraint_decode.py`, `geometry_decoder.py`, `partial_arrangement.py`, and `size_refinement.py`. The loader, architectures, neural weights paths, and training sources do not change between these tags. The release scorer evaluates both code versions against the same artifact tree. Thus describe v2 as **inference refinements over the same learned base**, rather than a new network trained end-to-end.

New in v2:

1. Suppress lower-confidence duplicate detections with centers <10 mm apart.
2. Correct learned region/gap bins when explicit sentence phrases disagree, retaining probability mass.
3. Replace partial movers by uniquely classified instances of the named color; v1 instead took the union of named-color instances and possibly erroneous learned movers.
4. Repair geometrically infeasible full/partial row layouts through selector-ranked partitions, checking at most eight returned proposals; partial proposals additionally must clear held objects.
5. Split mixed-color rows when the instruction requests single-color rows, color estimates have unique maxima, and the result is feasible.

**Already in v1:** RT-DETR+ROI appearance adapter, size residual, MiniLM/goal encoder, relation features, set attention, row/pointer heads, area ordering, original constrained beam search, geometry decoding/tightening/inset, partial obstacle translation, and **PCA silhouette orientation correction**. Do not claim these individually caused the measured v2 gain.

A concise manuscript comparison sentence: “Keeping the learned base fixed, v2 adds duplicate suppression, explicit region/gap consistency, stricter named-color selection, and feasibility-gated row/color repair.” The held-out result measures this bundle; it does not identify the marginal contribution of each addition.

## Compact mathematical loss definitions for the Appendix

Use `H_beta(e)=e²/(2 beta)` for `|e|<beta`, otherwise `|e|-beta/2`. Unless noted, `mean H` averages every active scalar element. Define `d_pi(a,b)=((a-b+pi/2) mod pi)-pi/2`.

**Size residual** (`experiments/size_refinement/run.py:391`):

`L_size = mean_{matched objects,k∈{major,minor}} H_.001(s_hat_ik-s*_ik)`.

Only size residual parameters update; detector query, predicted crop box and base sizes are detached. Matching is training-time source-center association, not an inference input.

**Selector** (`row_layout.py:373,380` plus selector saved logs):

`L_select = CE(row_logits, r*) + CE(pointer_logits, pi*)`.

Rows are supervised per active labeled object; pointer order is teacher forced during loss computation, with padded targets ignored. Inference uses predicted rows and predicted prefixes. There is a defined optional color-separation regularizer in source, but do **not** add it to the adopted selector objective without run evidence: recorded train_loss equals train_row+train_order.

**Restricted joint stage** (`joint_stage.py:159–178,342`):

`G={i : ||p_i^ref-p_i*||_2<.002 and |d_pi(theta_i^ref,theta_i*)|<3°}`, `F=valid\G`.

`L_joint=L_size + mean_{i∈F,k∈{x,y}}H_.010(p_hat_ik-p*_ik) + mean_{i∈F}|d_pi(theta_hat_i,theta*_i)| + mean_{i∈G,k∈{x,y}}H_.001(p_hat_ik-p^ref_ik)`.

Empty-set terms are zero. `s_hat` is stop-gradient before pose forward, all upstream representations are frozen, and the row angle head is frozen. The angular term is included in the scalar loss but cannot update the frozen angle head. Optimized pose parameters are row step and last origin layer only. This is not an end-to-end gradient through perception.

**Inherited relation training** (`relation_pose.py:137–163`, `run_relation_capacity.py:149`), optional if space:

`L_rel = mean_{batch,k} H_1((p_hat_first,k-p*_first,k)/.010) + mean_{valid neighbor pairs} sum_k H_1((Delta_hat_k-(p*_next,k-p*_prev,k))/.010) + mean_{active objects}|d_pi(theta_hat,theta*)| + CE(pointer_logits,pi*)`.

Note the subtle normalization: anchor averages coordinates; neighbor loss sums x/y per pair before averaging pairs. This training used teacher order, but final adopted inference reuses only the relation hidden encoder. The original relation displacement head is not the final v2 decoder.
