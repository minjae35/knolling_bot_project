# H/V Micro after execution-efficiency changes

The single authorized run **failed the full Micro Gate** after **173 optimizer
iterations / 57.462 driver seconds including terminal storage**, within the fresh
200 / 120-second cap. It completed the intended phase sequence. Neither head hit
its solver-time allowance; the remaining failure is XY precision, not starvation
by checkpoint I/O. No automatic retry, budget redistribution, other Micro, or
Oracle Layout run followed.

## Implementation and validation

Code commit: `c5b1966`. Architecture, losses, Dataset/Teacher, initial model weights,
seed, FP32 precision, and optimizer algorithms were unchanged. The implementation
and its compatibility limits are described in `recovery-v4-integrated-efficiency.md`.
**86 CPU/CUDA tests passed**, including real cached-model forward/backward without
optimizer steps and interruption/resume/rollback equivalence fixtures.

The existing Grounding-450 parent was hash-verified and its Pose tensors matched
Grounding update zero. A complete 561,415,535-byte checkpoint write/reload probe
passed. Approximately 13.13 GiB was available before execution, above the 12 GiB
preflight floor (18 possible full snapshots at 550 MiB plus 2 GiB reserve).
No checkpoint from either failed integrated attempt was used as a starting model.

The approved phase caps were 80/20/20/40/20/20. Each head received at most 10 seconds
of actual solver time, while all preparation, I/O, evaluations and solver calls
still counted toward 120 seconds. A 10-second finalization reserve was configured;
this run finished well before it was needed. The previous 25- and 89-update runs
remain separate historical attempts, not part of this fresh authorization.

## Phase results

| Phase | Updates | Wall time | Solver time | Maximum position | Maximum angle | Exit |
|---|---:|---:|---:|---:|---:|---|
| Coarse | 80 | 18.47 s | — | 17.016 mm | 89.892° | Phase update cap |
| XY calibration | 20 | 9.42 s | 0.253 s | 9.274 mm | 89.892° | Phase update cap; not converged |
| Angle calibration | 13 | 5.21 s | 0.108 s | 9.274 mm | 0.168° | True-bin probability criterion |
| Fine | 40 | 11.38 s | — | 10.370 mm | 0.171° | Phase update cap |
| Final XY calibration | 20 | 8.64 s | 0.213 s | 7.105 mm | 0.171° | Phase update cap; not converged |
| Final angle calibration | 0 | 3.25 s | 0 s | 7.105 mm | 0.171° | Already met probability criterion |

Final results versus the previous I/O-limited run:

| Metric | Previous | This run | Requirement |
|---|---:|---:|---:|
| Maximum position error | 17.124 mm | **7.105 mm** | <=1 mm |
| Maximum angle error | 89.892° | **0.171°** | <=2° |
| H/V change error | 89.966° | **0.203°** | <=4° |
| Minimum true-bin probability | 43.56% | **93.83%** | >=90% diagnostic |
| Independent Knolling validity | 0/2 | **0/2** | 2/2 |

Remaining geometry failures were vertical gap in both scenes and clearance in one.
There were no final collision or axis-alignment failures. Keep violations were
zero, but these two full-arrangement scenes do not establish Keep Micro performance.

Fine was entered once under the existing <=10 mm/<=2° condition. It retained
orientation but did not improve the worst positional error (9.274 -> 10.370 mm).
No previously valid scene/pair was lost and no keep violation increased, so existing
regression rules did not request rollback. Final calibration improved position to
7.105 mm but did not satisfy the Micro Gate.

The driver reports `budget_exhausted` because the phase schedule finished without
passing. This is **not** exhaustion of 120 seconds or all 200 global iterations:
angle stages used fewer updates after meeting their own criterion. The unused 27
iterations were not automatically reassigned to XY.

## Measured cost and the remaining optimization issue

Exclusive total time: solver **0.574 s**, Adam **5.896 s**, checkpoint I/O
**20.640 s**, immutable-file hashing **7.006 s**, feature-cache work **3.858 s**,
journal I/O **1.977 s**, evaluation **16.827 s**, other **0.684 s**. Feature-cache
work includes feature-producer hashing; it is not counted again in file hashing.
There were 117 line-search closure evaluations, 12 full checkpoints, six rollback
anchors, and only two feature caches. Angle phases reused XY features; Fine forced
a new cache before final calibration. All accepted head candidates were journaled.

The pipeline remains I/O-heavy in absolute terms, but overhead no longer prevented
any head from reaching its configured iteration cap or probability criterion.
Comparing raw run time with the previous run would be misleading: this run reached
all six phases, while the previous run stopped after the first three.

**Optimization was allowed to run, but XY did not converge.** Final XY Huber fell
from 0.28494 to 0.19279; the last five values were 0.195141, 0.193097, 0.192835,
0.192798, 0.192793. `converged` remained false, the final gradient norm was 0.2320
(maximum absolute component 0.04482), and no solver-time timeout occurred. Thus
20 iterations were not enough to demonstrate precision convergence for this
parameterization; elapsed compute time alone is not evidence of convergence.

Read-only CPU analysis of the final cached augmented feature matrix found:

- Four rows, rank four: a shared linear readout can represent arbitrary XY labels
  on these four points in exact arithmetic. This is a local capacity statement,
  not a generalization guarantee or a fitted replacement head.
- Singular values approximately **22.2316, 4.6936, 0.09152, 0.02647**; condition
  number **839.96**. All final coordinate residuals are below the 10 mm Huber scale,
  so the local problem is quadratic. Its nonzero per-coordinate Hessian condition
  number is approximately **705,531** under the equal sample weights here.
- **97.84%** of residual energy lies in the weakest singular direction, and another
  2.16% in the next weakest. Large errors remain in directions that change the
  objective slowly relative to dominant feature directions.

These observations strongly support poorly conditioned XY readout optimization
and incomplete convergence as the remaining issue. They do not establish that
more updates alone are the appropriate fix. A targeted next design review should
consider training-only XY preconditioning/scaling with weights mapped back to the
same inference head and the same loss, rather than changing the architecture or
repeating the unchanged run. No preconditioner, fitted diagnostic head, new
experiment, or additional optimizer step was implemented during this analysis.

## Preservation and artifacts

- Existing artifacts: **2,382 files** retained size, nanosecond modification time,
  inode and link count. Parent SHA256 remained
  `b1191dd1daaec602b366fba68cbd222c4031337130e4c981e0c2481bc01d0406`.
- All non-Pose model tensors, including Goal, MiniLM and Perception, exactly match
  the parent. Fine's optimizer reset count is one.
- Final checkpoint `axis/checkpoint-000173-0074.pt` loaded successfully; its terminal
  pointer matches. SHA256:
  `d033eca6b347c9454059d40e259fabf6a5ce3a01dca158a6028f7e44fec4c1d5`.
- Disk availability after the run was approximately **3.74 GiB**. No historical
  file was removed to obtain this space; another run would require a fresh storage
  check and sufficient capacity.
- tmux: `knolling-v4-integrated-hv-efficiency`.
  Reconnect: `tmux attach -t knolling-v4-integrated-hv-efficiency`.
- Run evidence: `artifacts/recovery_runs/recovery_v4_integrated_hv_efficiency_20261108T060406Z/`.
  `analysis.json` contains phase/time/loss traces and the matrix diagnostics;
  `axis/timing.json` records exclusive total and final-phase timings.
- Approval, config, storage probe evidence, launcher, console log and preservation
  baseline: `artifacts/recovery_launches/recovery_v4_integrated_hv_efficiency_20261108T060406Z/`.
