# Arrangement root-cause audit: confirmed mechanism, not a completed fix

**The extreme StructDiffusion XY error is caused by learned epsilon error being
amplified at the first reverse step under the selected near-zero-SNR cosine
endpoint.** A diagnostic intervention isolates this mechanism. This does not
explain every earlier Arrangement failure or establish that a schedule change
will achieve Task Success.

No production architecture, training code, configuration, Dataset, model weights,
checkpoint or `main` was changed. CPU diagnostics only; zero optimizer steps,
no GPU execution. A diagnostic-only dtype incompatibility was corrected in an
in-memory copy and the audit completed in 48.79 seconds. The initial error log
is retained rather than hidden. No neural fitting experiment was retried.

## Scope and evidence

- Selected 18k checkpoint from the one completed StructDiffusion experiment:
  `artifacts/structdiffusion_xy/checkpoint-018000-1794221040579648548.pt`.
- Official comparison: locally retained source pinned to
  `StructDiffusion/StructDiffusion@3d6dc8ac6723217b5550dac1db1b55c4d294d4bf`.
- 18,000 Train/Validation records audited for object-ID join, finite coordinates,
  workspace bounds and coordinate round trip. Test was not evaluated.
- Nine diagnostic Validation scenes, deliberately selected from saved results:
  two worst Full, one median-error Full failure, two worst Partial, one
  median-error Partial failure, and three passing Move scenes. This is an
  error-stratified diagnostic set, **not a new estimate of Validation accuracy**.
- Teacher target regeneration and physical validation on those nine scenes.
- All 200 reverse steps traced on the nine scenes. GT-noise interventions use
  labels exclusively inside the diagnostic; they are not inference solutions.

## Verdicts

| Check | Verdict | Evidence / boundary |
|---|---|---|
| Forward diffusion and reverse mean algebra | PASS | Upstream `q_sample` exactly matches; reverse mean agrees with independently derived posterior coefficients |
| Official default schedule reproduced | FAIL | Official active default is linear; current constructor hardcodes cosine |
| Selected cosine function reproduced | PASS | Beta tensors exactly match the pinned cosine implementation |
| XY units / normalization / ID pairing | PASS | 18,000 records; maximum coordinate round-trip error 0.0000298 mm |
| Teacher replay / physical labels | PASS | All nine selected scenes replay exactly and validate physically |
| True-noise end-to-end production sampler | PASS | Predicted-Move output maximum error 0.0000298 mm |
| Learned sampling stability | FAIL | First-step intervention reduces selected-set P95 346,520.75 -> 102.17 mm |
| CPU versus saved CUDA output tolerance | FAIL | 1/9 exceeds atol=0.0002 m, rtol=0.00002; maximum absolute difference 1.820 mm |
| Native `.double()` model execution | FAIL | Timestep embedding produces FP32 input for Double Linear; not a failure in the actual FP32 training run |
| Checkpoint / model / protected asset preservation | PASS | Model state and checkpoint/Dataset hashes unchanged; protected checks pass |
| Minimal overfit test | NOT RUN | Sampling stability failed and a cause was isolated; user required stopping further learning once a cause was confirmed |

CPU/CUDA tolerance failure must not be described as exact parity. It combines
device and batch-shape changes, whose contributions were not separately isolated.
Its <=1.82 mm discrepancy does not explain errors of tens/hundreds of metres.
The previous complete same-CUDA checkpoint reload was exact; this audit does not
contradict that result.

## Official formula and coordinate comparison

The local sampler follows the pinned `Sampler.sample` epsilon/DDPM equations:

```
z_t = sqrt(alpha_bar_t) z_0 + sqrt(1-alpha_bar_t) epsilon
mean = alpha_t^(-1/2) [z_t - beta_t epsilon_hat / sqrt(1-alpha_bar_t)]
z_(t-1) = mean + sqrt(posterior_variance_t) noise   (no noise at t=0)
```

The posterior mean was independently evaluated as a weighted sum of clean `z_0`
and `z_t`, not merely compared with a second copy of the epsilon formula.
Maximum discrepancy: **0.009245 mm in FP32**, **8.74e-12 mm in FP64**.

XY is in metres. Normalization is `(xy-[0.3,0.2])/[0.3,0.2]`; inverse is the exact
affine reverse. Dataset source/target ranges are x=[0.04008,0.55997] m,
y=[0.03348,0.36750] m. IDs join by `object_id`, not target rank. No factor-1,000,
XY swap or extra image-Y inversion was found. The 76,932 mm value is obtained
from genuinely huge model coordinates before evaluation, not a display-unit bug.

Adaptations that must not be confused with exact official inference reproduction:
2D observed attributes replace point clouds and 9D poses; predicted Keep states
remain fixed; a query-keyed fixed Gaussian stream replaces fresh draws per call;
there is no collision-discriminator selection, virtual structure transform or
geometric correction. The arithmetic corresponds to the basic sampler, not the
full `SamplerV2` pipeline.

## True-noise reconstruction

Two separate tests avoid confusing stochastic reverse noise with an inverse of
one forward draw:

1. Construct `z_t` from known `z_0` and epsilon at every timestep, then recover
   `z_0` algebraically using that same epsilon. Maximum coordinate error:
   **FP32 0.270683 mm**, **FP64 5.95e-10 mm**. FP32 cancellation is measurable at
   the extreme endpoint, but not remotely large enough to explain the failure.
2. Run the **actual production `sample()`** over all 200 steps. At each current
   state, diagnostic epsilon is `(z_t-sqrt(alpha_bar_t)*z_0)/sqrt(1-alpha_bar_t)`.
   It is recomputed after each stochastic transition; the original forward noise
   is not incorrectly reused for every step. Maximum predicted-Move coordinate
   error is **0.0000298 mm**. Keep routing stays unchanged; this result does not
   claim that GT was fed into real inference or that routing errors were repaired.

This separates sampler algebra from learned noise prediction. True noise works;
learned predictions fail. It does not prove neural learnability or Task Success.

## Causal isolation of the explosion

The official active schedule is linear (`noise_schedule.py:44`); cosine is a
commented alternative. This experiment explicitly selected cosine in
`structdiffusion_xy.py:61`, an adaptation introduced in the preceding work.
The endpoint has beta=0.99989998 and alpha-bar=6.07272e-9.

```
clean prediction error = -sqrt((1-alpha_bar)/alpha_bar) * epsilon error
```

The clean-state gain is **12,832.41**. The direct first reverse-step gain on
epsilon error is **99.9817**, versus **0.0216872** for the official default
linear schedule at 200 steps. This is ill-conditioning of the chosen prediction/
schedule combination, not floating-point overflow or a missing metre conversion.

With the same checkpoint, same observations, same noise bank and no optimizer:

| Nine-scene CPU diagnostic | Final XY P95, mm |
|---|---:|
| Learned predictions, FP32 | 346,520.75 |
| Higher-precision copy | 346,520.43 |
| True epsilon only at t=199; learned network at all later steps | 102.17 |
| True epsilon at every step | 0.0000197 |

Higher precision uses FP64 weights, attention and reverse arithmetic. The original
FP32 sinusoidal embedding is cast for the diagnostic Double Linear input; it is
not a claim of completely native FP64 inference. Maximum FP32/higher-precision
coordinate difference is 0.799 mm. This control does not eliminate every possible
precision effect, but it does not eliminate the catastrophic failure either.

In the free-running trace, normalized state RMS grows **1.135 -> 6.788 after the
first reverse transition -> 33.551 by t=190 -> 431.372 at t=0**. Initial epsilon
error RMS is 0.0682. The rollout leaves the normal scale of forward-noised training
states, and later denoising does not recover. Replacing only the first prediction
removes almost all of the catastrophic magnitude on these selected scenes.

The remaining error is real: after that intervention, the three Full failures
still have maximum object position errors **10.79, 10.61 and 7.61 mm**; the
median-error Partial failure still has **86.51 mm**. Therefore preventing the
explosion is necessary but does not establish correct gap/order/alignment.
These intervention scores are not deployable model results or a new gate pass.

## Actual scenes

- `expanded_1d2a5f455b80a6c9d4053776_1`: two-object, left-region, vertical,
  color-grouped Full command. Target centres approximately (0.150,0.144) and
  (0.150,0.255) m; predicted centres **(-166.977,-0.099)** and
  **(-348.299,-13.760)** m. Teacher replay and physical validity pass.
- `expanded_330aac8a5e3bd1403e2d3850_2`: eight-object Partial blue selection.
  Predictions stay near the workspace but fail collision/clearance/gap checks.
  First-step intervention still leaves 86.51 mm maximum error. This distinguishes
  ordinary layout learning failure from catastrophic rollout failure.
- Passing Full scene `expanded_43b067ec715a9e1b58a12198_1` has ~5.34 mm maximum
  Teacher-position error but satisfies the task evaluator. Teacher fidelity and
  Task Success are different criteria and must not be conflated.

## What the repeated earlier failures do and do not establish

Existing machine results were read, without reevaluating Test or retraining:

- Structure-centred model fit the old 76 Train scenes: **76/76, CF 35/35,
  XY P95 0.2273 mm**. Independent Validation was 10/38, all Keep-all successes.
  Blanket claims that the project cannot fit any task are false; this is not
  evidence of new-source generalization.
- Direct Centre–Local Train Full relative RMS decreased **23.716 -> 14.080 mm**
  from selected 1k to final 3k checkpoint. Corresponding saved Validation
  relative errors remain about **50.0 -> 49.3 mm**. Both incomplete fitting and
  a generalization gap remain; neither a capacity limit nor a single optimizer
  cause is established.
- Row/slot model: saved full-range structure evidence has **1,359/1,598 duplicate
  slot scenes**, exact structure 152/1,598 and selection accuracy **99.88%**.
  At the 1k small-scene checkpoint, Train Task 36/87 versus Val 1/85 and order
  accuracy 83.9% versus 41.8%. Correct selection is insufficient for layout.

These are different failure mechanisms. The current diffusion instability does
not retroactively explain the earlier deterministic decoders. Existing evidence
does not uniquely separate architecture capacity, optimization and distribution
effects behind their residual relational errors. No unverified global root
cause is presented as established.

## Smallest justified correction, not executed

**First restore the pinned upstream default linear schedule as an explicit,
checkpoint-validated training/sampling configuration, keeping the current neural
architecture unchanged.** The mathematical reduction in first-step amplification
is verified above; actual Task improvement is not verified. The current code
hardcodes cosine and ignores the configuration's schedule name when constructing
the module, so changing JSON alone is insufficient.

Do **not** switch a cosine-trained checkpoint to linear only at inference. It
was trained with a different timestep/noise distribution, and checkpoint buffers
can also silently overwrite constructor schedule values. Strict validation must
reject mismatched beta/alpha-bar arrays and schedule metadata. Preserve the old
checkpoint; this recommendation is not a migration that repairs its weights.

The source linear default also leaves terminal alpha-bar=0.13218, so its Gaussian
initialization approximation must remain an explicit limitation. Restoring a
known default is a minimal correction of the risky adaptation, not a guarantee
that the new layout problem is solved. No clipping, GT first-step substitution,
planner, new architecture or retrospective gate relaxation is proposed.

The code change is small; recovery of the existing model and generalization
success are **not established**. A later, separately authorized bounded small-set
learnability test would need to check real sampled target/task outputs, not only
teacher-forced epsilon MSE. It was not run here because the sampling stability
check already failed and the user instructed stopping further learning once a
cause was confirmed. Overfitting ability of this diffusion model remains unknown.

Evidence: `artifacts/structdiffusion_root_cause/audit.json`, `verdicts.json`,
`causal-trace.png`, `audit.log`, and `initial-dtype-check.log`. All 200 per-step
errors, scene commands/IDs/coordinates and intervention outcomes are retained.
Script: `scripts/audit_structdiffusion_root_cause.py`; launch only in `knolling`
with `CUDA_VISIBLE_DEVICES=''` and the existing offline HF/Torch cache settings.
It refuses to overwrite a completed audit.
