# recovery_v4_goal_joint — implementation review

**Implementation and forward/backward verification only. No Knolling model optimizer update or training run was performed.** Both v4 configs remain `review_only`, `allow_training: false`. AR/v3 remain available through their original configs and classes.

The v4-specific attachment was not present among the available files. This implementation follows the finalized in-conversation Goal Encoder + Single-stage Joint Pose specification. Work should reconcile any separately held final document before approving training.

Code reviewed by these checks: `d89b8a9` (with model/data implementation `3127c2f` and driver integration `6bcb800`).

## Design-to-code mapping

| Design | Code / contract |
| --- | --- |
| Frozen MiniLM + seven task queries | `recovery/goal_encoder.py`: 384→128 projection; seven learned language-attending slots; keep, axis, region, gap, group, sort heads; seventh referent slot participates in object grounding. |
| Object grounding | One `JointBlock`: object self-attention, cross-attention to all seven task slots, FFN, per-object selected logit. Source attributes only; no target-derived move labels. |
| Typed goal bottleneck | `g = [p_keep, (1-p_keep) p_axis, (1-p_keep) p_region, (1-p_keep) p_gap, (1-p_keep) p_group, (1-p_keep) p_sort]`, dimension 13. Pose receives this vector plus predicted object selection probabilities. No raw text or task-slot hidden-state bypass. |
| Single-stage joint poses | `recovery/goal_joint_layout.py`: source features (12) + selection (1), width 128, four heads, **two independently parameterized** set blocks, FFN width 512, dropout 0. Scene-shared FiLM conditions each FFN. FiLM weights use Xavier gain 0.1, biases 0. |
| Absolute target coordinates | `xy = [0.3, 0.2] + [0.3, 0.2] * raw_xy` in metres. No source addition, bounded residual, refinement, coordinate clamp or target-copy decision. |
| Orientation | Existing 18-bin pi-periodic representation. Final 38-channel head: XY (2), bin logits (18), residuals (18). |
| Input/output invariants | `RecoveryModel.forward(images, input_ids, attention_mask)` preserves source-query association. `GoalJointLayout.predict_layout` returns goal + layout through the same path used by training/inference. Object IDs are bookkeeping only. Null attention keys handle empty detections; invalid outputs are zero. |
| End-to-end gradients | Frozen MiniLM has no gradients. Frozen Goal Encoder **does not use no_grad/detach** around its forward: selected-goal dependence on source features can still transmit gradients to Perception during Joint tuning. |

Only offline label preparation and independent evaluation use instruction/Teacher metadata. Production inference does not invoke a Teacher, planner or GT count/goal/target.

## Loss and freeze policy

Goal loss has three equally weighted groups:

```
L_goal = BCE(keep) + (CE(axis)+CE(region)+CE(gap)+CE(group)+CE(sort))/5 + BCE(selected)
```

Attribute CEs average applicable scenes; keep-all scenes mask all five attributes. Selection BCE averages valid objects per scene, then scenes. `keep_all` is not inferred from `mode=no_op`; selected objects are not inferred from target movement. Gap vocabulary is exactly 15/20/30 mm; unsupported active values fail rather than round.

Pose reuses the three final-only loss terms:

```
L_pose = mean_scene mean_valid sum_xy SmoothL1((pred_xy - target_xy) / 0.010)
       + mean_scene mean_valid CE(angle_logits, target_bin)
       + mean_scene mean_valid SmoothL1(tanh(residual[target_bin]), target_residual)
```

All valid objects, including fixed/keep objects, are supervised. No Move/Keep pose classifier, geometry/relation loss or intermediate loss is added.

| Phase | Trainable modules | Inputs / acceptance |
| --- | --- | --- |
| Grounding | Goal Encoder + object grounder | GT source objects + text; full 76-record grounding gate. Pose and Perception frozen. |
| Micro Pose | Pose only | Three separate fixed diagnostic subsets; predicted frozen Goals. Each starts from the **same passed Grounding checkpoint**, including its original Pose weights. |
| Oracle Layout | Pose only | All 76 records, predicted frozen Goals; starts from Grounding, never from a Micro-trained Pose. All three passed diagnostics required. |
| Perception | Existing detector/source heads | Existing source supervision and source gate; Goal/Pose frozen. |
| Predicted Layout | Pose only | Actual inference-selected objects; source-only matching attaches targets after selection. Bad correspondence aborts. |
| Joint | Pose + existing late Perception subset | Predicted-source layout plus separate unique-source detection replay. Goal weights stay frozen; regression triggers rollback. |

V4 instruction epochs visit every record once, shuffled, without legacy mode oversampling. Perception still uses unique source families. MiniLM/RGB caches remain optional; learned Goal predictions are never cached.

## Data and gates

The existing `continuous_rows_v3` Teacher, RGB and 76 annotations are unchanged. `GoalDataset` adds flat supervision tensors in memory; labels cannot enter `Batch.inputs`. Source/Teacher validation remains in `RecoveryDatasetV3`.

- Existing annotation SHA256: `b42ee5b9e99afeeb37036a63f4e180b791719212faa68be226f7745beb051e7c`.
- Goal dataset fingerprint: `3af9210f8cdb5d9acda5084200940c4d2407c6a28a7268cc36f55c4793b7e6bf` (parent fingerprint + `recovery_goal_labels_v1` + ordered vocabulary).
- Scope: 76 Train verification records, 25 source families, 35 counterfactual pairs covering six factors. This is **not held-out/generalization evidence**.
- Grounding: every applicable global attribute and valid object's selection must be correct, with true-class probability ≥0.90. Both members of all 35 contrasts must satisfy their entire semantic label vector, covering changed and unchanged fields. Missing/duplicate records, incomplete contrasts and nonfinite probabilities fail.
- Micro: maximum object position error ≤1 mm, maximum angle error ≤2°; independent geometry/semantics including keep ≤1 mm/1°. The axis pair additionally requires relative angle-change error ≤4°.
- Full Oracle Layout: grounding passes, all 76 layouts valid, all 35 pose contrasts pass. Teacher fidelity (P95 ≤1 mm / 2°) is reported separately; alternative valid layouts do not fail solely for differing from canonical Teacher coordinates.
- Connected stages: existing source gate + full goal/layout validity; Joint additionally checks full-set regression. Grounding confidence may vary above its acceptance threshold; it need not increase monotonically.

Fixed Micro sample IDs (`recovery/evaluation_v4.py`):

| Case | Sample IDs |
| --- | --- |
| Large translation | `v3_00f5b6400c9c805edd26f2f6` |
| Keep-all | `v3_1f2cb71631ace3774bd5e1b6` |
| Axis counterfactual | `v3_1efbc2cf2648c75dd2dcca9f`, `v3_0c1e24b0cf284df95555f234` |

## Driver, resume and review boundary

`driver.py` retains bounded updates/time, pause/resume, final timeout evaluation, Joint regression rollback and exclusive checkpoint publication. New checkpoints additionally bind architecture, v4 evaluator, goal schema/vocabulary, Grounding-parent SHA256, diagnostic/production purpose, Micro case and evidence. Strict loading rejects incompatible legacy/model/data/config/code contracts.

`python -m knolling_model.recovery.micro` is the future three-case coordinator. It starts every case from the same Grounding file and stops on the first failure/pause. Individual diagnostics resume via the normal driver with `--stage micro_pose --micro-case ... --resume ...`. Oracle Layout requires the Grounding checkpoint via `--parent-checkpoint` and three passed diagnostics via `--micro-checkpoints`; diagnostic model weights are never loaded into production.

The new review config contains **proposed, not approved** ceilings: Grounding 500 updates/600 s; each Micro case 200/120 s; Oracle Layout 500/600 s. Later-stage ceilings retain the existing driver values. They are placeholders for independent review and explicit training authorization. Neither old approvals nor the old AR budget-extension path authorize v4 training. The CLI requires a new receipt bound to the reviewed code/config/data/stage.

## Verification evidence

All Python/tests used Conda `knolling`, PyTorch `2.9.1+cu128`. Local logs and preservation evidence: `artifacts/recovery_v4_implementation_verification_20261108/`.

| Check | Result |
| --- | --- |
| Static Python parsing | 123 files passed; `git diff --check` passed. |
| Whole repository CPU checker, including cached-model checks | **262 passed, 27 skipped**, 85.78 s. Skips are explicitly gated CUDA, rendering/migration and other optional integration checks. |
| Subsequent driver/coordinator/evaluation checks | **15 passed, 1 CUDA skip**, 29.04 s. Covers the added coordinator and final confidence-regression policy. |
| CUDA-enabled v4 model/driver/runtime suite | **24 passed**, 25.91 s. Includes CPU cases and explicit CUDA cases. |
| Final nonfinite-goal regression | **6 passed**, 12.93 s; no model updates. |
| Preservation | All **2,252** pre-existing artifact files retained with unchanged size and nanosecond modification time. Annotation SHA remains pinned. |

Coverage includes 1–8 object permutations, ID independence, object/text padding and empty detections, absolute unclamped coordinates, frozen-Goal input gradients, stage freeze/group disjointness, GT corruption invariance, training/inference output identity, strict state roundtrip and legacy rejection, uniform data exposure, exact-output evaluator fixtures, missing/duplicate cases, confidence gates, paused-vs-uninterrupted scalar optimizer/RNG/cursor identity, Micro-parent isolation, and CPU/CUDA timeout rollback.

Real cached RT-DETR R18 + MiniLM on RTX 4090, batch 4, no optimizer steps:

| Phase | FP32 peak MiB | BF16 peak MiB |
| --- | ---: | ---: |
| Grounding | 624.30 | 621.42 |
| Micro Pose | 620.44 | 619.79 |
| Oracle Layout | 620.03 | 619.79 |
| Perception | 2846.42 | 2589.01 |

Real RGB inference was finite. Before/after state dictionaries were exactly equal. Connected predicted-layout/Joint backward and image gradients were also checked with the tiny RT-DETR fixture. Peak 2846 MiB is approximately **2.78 GiB**; these are verification measurements, not a batch-size throughput benchmark.

Scalar fixtures alone perform optimizer updates to test lifecycle behavior. Annotation-backed perfect-output fixtures test evaluator correctness and are never used by production inference; their passing scores are **not model performance**.

## Work's independent review priorities

1. Confirm the typed bottleneck, seven slots, two independent FiLM set blocks and absolute-coordinate head match the separately held final design.
2. Check Goal label semantics, keep/no-op distinction, confidence gate and Counterfactual completeness. Review the strict 100%/0.90 requirement before assigning a training budget.
3. Inspect Micro checkpoint provenance and the production restart from original Grounding Pose weights; verify timeout/resume/rollback with the v4 stage order.
4. Confirm frozen Goal gradients reach predicted source attributes without GT-assisted query selection or a text/label bypass.
5. Approve the actual phase budgets before any learning. **Learnability, Knolling success and generalization remain unmeasured.** Absolute predictions may initially leave the workspace; evaluation rejects this without rule-based repair. Sparse contrast coverage and only 25 families remain data limitations.
