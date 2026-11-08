# Integrated calibration execution efficiency

The H/V restart stopped after five XY and four angle updates because each
10-second head wall-time allowance included checkpoint/cache preparation. This
change retains v4 forward mathematics, losses, optimizer algorithms, phase update
caps, and the pending-journal decision ordering.

## Execution changes

- Hash each immutable full journal base once per process, reuse that digest only
  while device/inode/size/mtime/ctime match, and reject mutations. Restart/load
  still recomputes content hashes; no persisted digest is trusted without checking.
- Replace the second full checkpoint after feature preparation with a small atomic
  journal. The phase-entry base, rollback anchor, feature artifact, solver state,
  cursor and RNG remain recoverable before any accepted solver update.
- Reuse the feature artifact and in-memory cache across head_xy -> head_angle and
  final_xy -> final_angle only. Shared/residual weights do not change on those
  transitions. Fine invalidates the cache, and final calibration captures new
  features. A restarted process still validates cache hash/producer/data/mapping.
- Every accepted candidate is still journaled before evaluation/observers; resume
  resolves pending evaluation, Gate, regression and phase completion before another
  update. No asynchronous or deferred durability is introduced.

## Time policy

Existing configs retain the original `head_time_mode: wall` default. The authorized
H/V efficiency experiment explicitly uses `head_time_mode: solver`, retaining
`head_seconds: 10` and phase caps **80/20/20/40/20/20**, total **200 / 120 seconds**.

Solver seconds count actual L-BFGS calls, including line-search closures and
failed/aborted trials, with CUDA synchronization for accurate wall timing. These
seconds survive checkpoints/journals and resume. Preparation does not consume the
solver allowance, but **all measured work still consumes the global wall budget**.
`finalize_reserve_seconds: 10` stops new work at 110 seconds, reserving time for
required final evaluation/rollback and durable terminal storage. It does not add
time to the 120-second budget. There is no phase or global automatic extension.

Exclusive cumulative and per-phase counters distinguish `solver`, `adam`,
`feature_cache`, `hashing`, `checkpoint_io`, `journal_io`, `evaluation`, and `other`.
Nested timers are charged once; hash CPU work internal to feature construction
belongs to `feature_cache`. The initial rollback anchor now counts in driver time.
Phase results and checkpoint/journal state contain timers; `timing.json` records
completion including the terminal checkpoint's write. Timing-report failure cannot
discard the training result. Model loading before engine execution is outside the
existing driver budget, as before. Timer snapshots precede their own durable write;
an abrupt process loss cannot account for wall time after the last durable record.

The budget is enforced at safe computation boundaries, with time reserved for
finalization; a stalled filesystem call cannot be preempted without jeopardizing
durable storage. Actual completion time must therefore be reported, not assumed.

## Verification scope

Tests cover CPU/CUDA phase and journal interruption; identical update counts,
weights, optimizer and RNG state; pending PASS/regression/convergence/cap decisions;
cache preparation interruption; cache invalidation after Fine; corrupt base/cache
rejection; synthetic I/O overhead; persisted solver-time allowances; global
finalization reserve; and nonfatal timing-report failure. The real cached model
also undergoes CUDA forward/backward and readout/inference checks without optimizer
steps. No architecture, loss, annotation, historical checkpoint or existing result
is modified.

Validation: **86 passed in 55.60 seconds**, using `knolling` with
`RUN_RECOVERY_CUDA_TESTS=1`, offline cached Hugging Face/Torch assets, and the
integrated, integrated_resume, integrated_efficiency, readout and provenance test
modules. `git diff --check` passed. Initial test-only failures (synthetic phase
expectations, CPU comparison of a CUDA fixture, and a missing cache environment
variable) were corrected before the complete passing run.
