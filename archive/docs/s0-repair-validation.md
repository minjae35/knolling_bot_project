# Explicit objectness policy and bounded repair candidates

The original configs retain `legacy_query_weight`: matched queries have weight 1,
unmatched queries weight 0.1, divided by total query weight. At Q=300 and N=2,
foreground has only 6.3% of the aggregate label weight. This is an observable
capacity dependence, not proof that this alone caused the S0 failure.

An opt-in `objectness_normalization: "balanced_groups"` computes the existing
binary cross-entropy separately over positives and negatives, then averages
the two group means. Duplicating empty queries therefore cannot dilute the
total foreground weight. If all queries are positive, use the positive mean.
The no-positive case is rejected because existing S0 scenes are nonempty.

This preserves the full pretrained 80-class classifier, max-class objectness,
threshold 0.5, Source Hungarian assignment, physical losses, four-stage Detection
mean and Target losses. It introduces no model parameters, filtering, planner
or GT input to the neural forward. It is **experimental and disabled in existing
configs**. Passing analytic tests does not establish correct count/duplicate
learning, retention or S0 convergence; failed micro candidates must not be
adopted as a training fix.

The criterion factory is used consistently by the training runner and GPU
preflight. Provenance records the policy, and strict resume rejects a changed
policy. Historical provenance with no policy field is interpreted as legacy
when all other provenance checks match. Source revision checks remain strict.
Changing the objective for a disposable experiment is an explicit weight-only
warm start with a fresh optimizer, not resumption of the old experiment.

## Controlled repair sequence

`scripts/run_s0_repair_candidate.py` consumes a separate immutable root protocol
and starts one named attempt. It refuses an existing attempt directory and a
base-checkpoint hash mismatch. The gates remain in `s0_validation_gates.py`.

- Detection1: balanced group BCE, final classifier only, at most 100 updates.
- Detection2: only after Detection1 failure; same supervision, permit the
  existing perception/physical Source parameters to learn, at most 100 updates.
- Language1: only after a Detection gate passes; existing paired instruction
  Target mean, predicted Source, train text projection and Arrangement only.
- Joint: only after both bottleneck gates pass, at most 200 full forward/backward
  updates, equal family weights and the predeclared all 32 admission criteria.

A failed second candidate stops the entire sequence. A further language repair
must be implemented and reviewed against its actual first-candidate evidence
if that stage is reached; this tool does not advertise an unimplemented text
unfreezing path. No automatic full S0 or S1 command exists here.

Every micro starts from the same original weights with fresh AdamW, configured
LRs, seed 1203, strict determinism and fixed threshold 0.5. Cached features are
used only for explicitly frozen perception. Trainable perception uses actual
image forwards. Snapshots contain optimizer/RNG/provenance and cursor state at
midpoint/final evaluations for interruption review. There is no automatic fresh
restart of an interrupted attempt, which could silently exceed its budget.

Diagnostic checkpoints use `s0_repair_candidate_v1`, which the official S0
resume loader rejects. Model tensor shapes remain compatible with the original
checkpoint. Preserve original data/reports and keep all diagnostic weights,
protocols, metrics and internal assessments under ignored `artifacts/`.

See [the admission thresholds](s0-bounded-validation.md). Loss reduction, local
training-set success or count accuracy alone cannot waive TP/FN/FP, confidence
margin, all 32 retention, pose or language gates. Telegram registration belongs
to the entire experiment; only publish its terminal result after all authorized
work is complete, using the existing managed dispatcher.
