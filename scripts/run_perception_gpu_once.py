"""Exactly one registered production run after a passing zero-update CUDA check."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from knolling_dataset.builder import file_hash
from knolling_model.perception import checkpoint
from knolling_model.perception.pipeline import gpu_guard, train, write_json
from run_recovery_stages import register_terminal_run


def run(config_path, preflight_path, output):
    if Path(sys.prefix).name != 'knolling':
        raise RuntimeError('use the existing knolling environment')
    config = json.loads(Path(config_path).read_text())
    pins = json.loads(Path(config['pretrained']).read_text())
    preflight = json.loads(Path(preflight_path).read_text())
    if (preflight.get('passed') is not True or preflight.get('optimizer_updates') != 0
            or preflight['config_sha256'] != file_hash(config_path)
            or preflight['model_code'] != checkpoint.model_provenance()
            or preflight['code'] != checkpoint.code_provenance()
            or config['max_updates'] != 500 or config['max_seconds'] != 1800
            or config['batch_size'] != 2 or config['evaluate_every'] != 50):
        raise ValueError('CUDA preflight or fixed single-run contract mismatch')
    gpu_guard(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Durable once-only run claim, no retries.
    receipt = dict(schema='perception_single_run_receipt_v1',
        config_sha256=file_hash(config_path), cuda_preflight_sha256=file_hash(preflight_path),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        authorization='User explicitly requested exactly one GPU Perception run after CUDA preflight',
        max_updates=500, max_seconds=1800, automatic_retry=False)
    write_json(output/'launch-receipt.json', receipt)
    final = dict(status='failed', stop_reason='execution_error', automatic_retry=False)
    try:
        register_terminal_run(output, config['max_updates'])
        result = train(config, pins, output/'training')
        # Match the managed terminal dispatcher's public completion schema. No S0
        # or demo success flag is asserted by merely reaching a run budget.
        final = dict(status='completed', stop_reason='update_budget',
                     actual_stop=result['stop'], updates=result['updates'],
                     checkpoint=str((output/'training/best.pt').resolve()),
                     automatic_retry=False, training=result)
        return result
    except BaseException as error:
        final.update(error=type(error).__name__, message=str(error))
        raise
    finally:
        write_json(output/'result.json', final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cuda-preflight', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.cuda_preflight, args.output), indent=2), flush=True)
