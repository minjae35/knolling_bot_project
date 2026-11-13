# v4 integrated H/V Micro — storage failure before calibration

The single authorized run on **2026-11-08 05:39 UTC** stopped during Coarse at
**25 optimizer iterations** because the 100 GB root/workspace filesystem filled.
The first periodic checkpoint raised `OSError: [Errno 28] No space left on device`;
the error-path rollback checkpoint also could not be saved. This is an
**infrastructure failure**, not a completed test of the integrated optimization
strategy. No retry, extension, other Micro or Oracle Layout run was launched.

## Execution and provenance

- Code: `33445390fb45504a7b5f17c92cd70366fc75302e`, `feat/neural-recovery`.
- Environment: `knolling`, RTX 4090, FP32 per the reviewed recipe, cached frozen
  text and the two fixed H/V records, existing tqdm progress and ETA.
- Budget: total 200 optimizer iterations / 120 seconds, phases
  80/20/20/40/20/20; each head solver additionally capped at 10 seconds.
- Parent: `recovery_v4_grounding_20261108T041737Z/grounding/checkpoint-000450-0019.pt`,
  SHA256 `b1191dd1daaec602b366fba68cbd222c4031337130e4c981e0c2481bc01d0406`.
- The parent's Pose weights were verified identical to Grounding update 0;
  no previous Micro-trained Pose weights or optimizer state were reused.
- Initial Pose tensor hash:
  `6af18bc92b5eab7b340b32fb6212163ec7f2050d94f364eda7c63e042ff2b50a`.
- Explicit Grounding transfer code/config/data contracts passed preflight.
- tmux: `knolling-v4-integrated-hv`; reconnect with
  `tmux attach -t knolling-v4-integrated-hv`. The stopped pane retains its output.

## Last completed evaluation, update 25

These are logged predictions, **not a successfully saved 25-update model**.

| Metric | Result | Micro requirement |
|---|---:|---:|
| Maximum position error | 36.004 mm | <=1 mm |
| Maximum angle error | 89.952 degrees | <=2 degrees |
| H/V change error | 89.972 degrees | <=4 degrees |
| Independent Knolling validity | 0/2 scenes | 2/2 |
| Keep violations | 0 | 0 |
| Goal axis classification | 2/2 | correct both |
| Goal selection | 4/4 | correct all |

Goal's minimum true-label probability was 0.993934. However, Pose orientation
changed only about 0.0343 degrees for object o01 and 0.0276 degrees for o02
between H/V, versus the required 90-degree change. Both outputs remained near
horizontal. Position improved from the initial maximum 422.956 mm to 36.004 mm;
training loss decreased from 43.3618 at update 1 to 4.0356 at update 25.
The last completed evaluation was at 5.493 cumulative driver seconds.

## Failure attribution and preservation

- **Coarse:** began normally; stopped by storage failure at its checkpoint write.
  Twenty-five updates are insufficient evidence to diagnose representation or
  optimization failure under the planned 80-update Coarse budget.
- **Head calibration and Fine:** never entered. Their efficacy remains untested.
- `anchor.pt` and `checkpoint-000000-0001.pt` are the only weight artifacts from
  this run. Each is approximately 532 MiB; neither contains the 25 trained updates.
  The log records those predictions, but they cannot substitute for a checkpoint.
- The filesystem had approximately 13 MiB free after failure. Free storage was
  not checked before launch; that preflight omission must be corrected before any
  future run. The full recipe stores multiple full model/phase snapshots and
  requires substantially more space than this initial pair of files.
- All **2,390 pre-existing artifact files** retained their size and nanosecond
  modification time. No dataset/checkpoint/result was deleted or overwritten.
- No budget-exhaustion/resume guard was bypassed. A future rerun must account for
  the 25 already executed updates; this run was not automatically restarted.

Run evidence:
`artifacts/recovery_runs/recovery_v4_integrated_hv_20261108T053906Z/` contains
`result.json`, `analysis.json`, `axis/history.jsonl` and the initial weight files.
Configuration, exact approval receipt, transfer hashes, launcher, console trace
and preservation baseline are under
`artifacts/recovery_launches/recovery_v4_integrated_hv_20261108T053906Z/`.
The existing managed notification dispatcher was registered for this run; no
periodic notification worker or direct Telegram sender was started.
