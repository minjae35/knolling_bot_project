#!/usr/bin/env python3
"""Run approved recovery stages once; extensions require an explicit source grant."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from knolling_model.recovery.driver import read_config
from knolling_model.recovery.stages import STAGES


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def register_terminal_run(output, total):
    """Register with the existing notify.py dispatcher; no direct sends/workers."""
    path = Path('/opt/knolling-notify/watch.json')
    previous = json.loads(path.read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    proc = Path(f"/proc/{previous['pid']}/stat")
    if proc.exists() and previous['boot_id'] == boot:
        fields = proc.read_text().rsplit(') ', 1)[1].split()
        if fields[0] != 'Z' and fields[19] == previous['start_ticks']:
            raise RuntimeError('another live experiment owns notification registration')
    write(output/'previous-notification-registration.json', previous)
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    record = dict(run_dir=str(output.resolve()), pid=os.getpid(), start_ticks=fields[19], boot_id=boot, total=total)
    write(output/'notification-registration.json', record)
    temporary = path.with_suffix('.recovery.tmp')
    with temporary.open('x') as stream:
        json.dump(record, stream)
    temporary.chmod(0o600)
    temporary.replace(path)


def run_stages(config_path, approval, output, *, register=False, execute=subprocess.run, extend_from=None, stop_after='joint'):
    config = read_config(config_path)
    if stop_after not in STAGES:
        raise ValueError('unknown final authorized stage')
    requested = STAGES[:STAGES.index(stop_after)+1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, parent = {}, None
    try:
        if register:
            register_terminal_run(output, sum(config['stages'][s]['max_updates'] for s in requested))
        for stage in requested:
            destination = output/stage
            command = [sys.executable, '-u', '-m', 'knolling_model.recovery.driver',
                       '--config', str(config_path), '--approval', str(approval),
                       '--stage', stage, '--output', str(destination)]
            if parent:
                command += ['--parent-checkpoint', parent]
            elif extend_from and stage == 'oracle_layout':
                command += ['--extend-from', str(extend_from)]
            print(f'\nStarting {stage}: {config["stages"][stage]}', flush=True)
            process = execute(command, check=False)
            path = destination/'result.json'
            if not path.is_file():
                raise RuntimeError(f'{stage} exited {process.returncode} without a final result')
            result = json.loads(path.read_text())
            results[stage] = result
            write(output/f'{stage}-summary.json', result)
            if process.returncode != 0 or result['status'] != 'passed':
                break
            parent = result['checkpoint']
        success = len(results) == len(requested) and all(r['status'] == 'passed' for r in results.values())
        last = next(reversed(results.values()))
        final = dict(status='passed' if success else 'stopped', stages=results,
                     stop_reason='memorization_gate' if success else (
                         'update_budget' if last['status'] == 'budget_exhausted' else 'stage_failure'),
                     s0_gate=success and stop_after=='joint', requested_stages=list(requested),
                     requested_stages_passed=success, automatic_extension=False)
    except Exception as exc:
        final = dict(status='failed', stop_reason='execution_error', error=f'{type(exc).__name__}: {exc}',
                     stages=results, s0_gate=False, automatic_extension=False)
    write(output/'result.json', final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--register-terminal-notifications', action='store_true')
    parser.add_argument('--extend-from')
    parser.add_argument('--stop-after', choices=STAGES, default='joint')
    args = parser.parse_args()
    if Path(sys.prefix).name != 'knolling':
        raise RuntimeError('use the knolling Conda environment')
    result = run_stages(args.config, args.approval, args.output, register=args.register_terminal_notifications,
                        extend_from=args.extend_from, stop_after=args.stop_after)
    print(json.dumps(dict(status=result['status'], stop_reason=result['stop_reason'], output=args.output)), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 2)


if __name__ == '__main__':
    main()
