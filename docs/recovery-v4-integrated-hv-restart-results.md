# v4 integrated H/V Micro restart — head time limits reached

The single fresh restart on 2026-11-08 **did not pass H/V Micro**. It stopped as
`fine_ineligible` after **89 optimizer iterations**, with 38.262 driver seconds at
the terminal checkpoint request and **39.476 seconds after terminal save**.
No retry, extension, other Micro, Fine, Final Calibration, or Oracle Layout was run.

## Authorization and validated starting state

- The user explicitly granted a **fresh 200 iterations / 120 seconds** for this
  restart; the previous storage-failed run's 25 updates are separately recorded.
  Combined optimizer work across these two attempts is 114 updates, not 89.
- Phase caps remained **80/20/20/40/20/20**, with each head phase capped at 10 seconds.
- Execution commit: `687615e`. Its implementation, configuration files, and tests
  are unchanged from Work-reviewed `3344539`; the intervening change is documentation.
- Environment: `/venv/knolling`, RTX 4090, reviewed FP32 recipe, two fixed H/V
  records, cached frozen text, and tqdm progress inside tmux.
- Parent: `recovery_v4_grounding_20261108T041737Z/grounding/checkpoint-000450-0019.pt`.
  SHA256: `b1191dd1daaec602b366fba68cbd222c4031337130e4c981e0c2481bc01d0406`.
  Goal passed at update 450; Pose tensors equal Grounding update zero exactly.
- No failed, partial, or trained Micro checkpoint was used. The earlier missing
  update-25 checkpoint was neither reconstructed nor treated as a resume source.
- Preflight required at least 18 GiB free; approximately 20.70 GiB was available.
  The production atomic writer saved and reloaded a **561,415,535-byte** complete
  checkpoint, verifying identical model tensors and the original file SHA256.
  Only that newly created write probe was removed after verification.
- tmux: `knolling-v4-integrated-hv-restart`.
  Reconnect: `tmux attach -t knolling-v4-integrated-hv-restart`.

## Actual results

| Phase | Accepted iterations | Maximum position error | Maximum angle error | Exit |
|---|---:|---:|---:|---|
| Initial evaluation | 0 | 422.956 mm | 89.804° | Gate failed |
| Coarse | 80 | 17.016 mm | 89.892° | Planned phase iteration cap |
| XY calibration | 5 | 17.124 mm | 89.892° | `head_time_budget` |
| Orientation calibration | 4 | 17.124 mm | 89.892° | `head_time_budget` |
| Fine / Final calibration | 0 | — | — | Ineligible; not executed |

Final Micro requirements and outcomes:

- Maximum position **17.124 mm**, requirement <=1 mm.
- Maximum angle **89.892°**, requirement <=2°.
- H/V change error **89.966°**, requirement <=4°; meaningful 90° command-dependent
  orientation separation was not achieved.
- Independent Knolling validity **0/2**, keep violations 0 (these are full-arrangement
  scenes, so this is not a Keep Micro validation).
- Four true-bin probabilities: **49.45%, 43.56%, 47.34%, 53.56%**; calibration
  requires every value >=90%.
- Goal remains correct on this H/V subset; minimum true-label probability 99.393%.
  This subset run is not a new full 76-record/35-pair Grounding evaluation.
- Global budget remained available, but both head phase time limits were reached.
  The full-set <=10 mm / <=2° Fine eligibility condition then correctly blocked Fine.
  No successful scene or pair had been lost, so the terminal state was not rollback.

## Failure attribution: evidence and limits

**The observed calibration stop was wall-clock limited, not convergence or the
20-iteration head cap.** XY accepted five L-BFGS updates and orientation four,
with 25 closure evaluations total. Orientation CE decreased from 0.973 after its
first accepted update to 0.727 after its fourth; that is not successful separation.
XY's maximum error varied around 16.7–18.9 mm and ended at 17.124 mm.

The timer begins before `set_phase` saves its full anchor/checkpoint. It also
includes `prepare_head` feature capture, full-model feature hashing/cache
validation, and another full checkpoint. From the preceding phase's end to the
head-ready checkpoint event, this consumed approximately **6.2 seconds for XY**
and **6.6 seconds for orientation** out of each 10-second phase cap. Accepted
candidate events were then about 0.9 seconds apart. `journal()` also hashes its
approximately 0.5 GiB immutable base checkpoint for each candidate.

Thus persistence and preparation materially reduced time available for fitting.
The recorded timings combine I/O, hashing, feature capture and validation; they
do not isolate each operation's exact cost. No extra profiling run or optimizer
step was performed. The source locations are `IntegratedEngine.set_phase`,
`prepare_head`, `journal`, and `check` in `src/knolling_model/recovery/integrated.py`.

This does **not** establish that more solver steps would pass XY or orientation,
or that features after 80 Coarse updates are as separable as the earlier diagnostic's
features after 200 updates. It also does not establish architectural impossibility.
The strongest next implementation target, before requesting another experiment,
is reducing repeated checkpoint/hash overhead while preserving durable candidate
journals, Resume correctness, and the approved wall-time accounting. No such code
change or budget extension was made here.

## Artifact and preservation verification

- Final checkpoint: `axis/checkpoint-000089-0022.pt`, SHA256
  `36425ef8485484830660b749402754f00b357523e5ed35048cbcc392a64c1a33`.
- Final checkpoint CPU reload and `terminal.pt` pointer validation succeeded.
- Comparing Coarse-end and final model tensors shows changes only in XY and
  orientation-logit rows; shared features and residual rows remained identical.
- Computing the saved XY readout on the saved feature cache reproduces the final
  evaluation maximum within 0.001 mm. Saved orientation probabilities match the
  final head diagnostic. No solver ran during this read-only verification.
- **2,344 pre-existing artifact files** retained size, nanosecond modification
  time, inode and link count. Parent SHA256 remained unchanged. No old dataset,
  checkpoint, result, or hard-link arrangement was changed.
- Final disk availability was approximately **14.44 GiB**. No storage error or
  observer error occurred. Existing managed terminal notification registration was
  reused; no progress notification worker or direct message transport was created.

Run artifacts and analysis:
`artifacts/recovery_runs/recovery_v4_integrated_hv_restart_20261108T055411Z/`.
Approval, configuration, write preflight, launch command, console trace, and
preservation baseline:
`artifacts/recovery_launches/recovery_v4_integrated_hv_restart_20261108T055411Z/`.
