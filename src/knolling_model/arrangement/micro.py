"""Fixed micro diagnostics; each run must independently load its grounding parent."""
from .evaluation_v4 import MICRO_CASES


class MicroDataset:
    def __init__(self, dataset, case):
        if case not in MICRO_CASES:
            raise ValueError('unknown micro case')
        ids = [r['sample_id'] for r in dataset.records]
        self.indices = [ids.index(s) for s in MICRO_CASES[case]]
        self.dataset = dataset
        self.records = [dataset.records[i] for i in self.indices]
        self.fingerprint = dataset.fingerprint

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def main():
    """Future approved use only: stop at the first failed/paused micro case."""
    import argparse
    import json
    from pathlib import Path
    import subprocess
    import sys
    from .driver import read_config
    from .goal_joint_layout import ARCHITECTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/recovery_v4_execution.json')
    parser.add_argument('--grounding-checkpoint', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    if config.get('allow_training') is not True:
        raise PermissionError('training disabled; micro diagnostics require independent review')
    if config['model']['architecture'] != ARCHITECTURE:
        raise ValueError('micro coordinator requires v4')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('micro coordinator requires a new output directory; resume individual cases with the driver')
    checkpoints = []
    for case in MICRO_CASES:
        folder = output/case
        # Each subprocess enforces the full receipt/code/config/data contract.
        subprocess.run([sys.executable, '-m', 'knolling_model.arrangement.driver', '--config', args.config,
                        '--stage', 'micro_pose', '--micro-case', case, '--parent-checkpoint', args.grounding_checkpoint,
                        '--approval', args.approval, '--output', str(folder)], check=True)
        result = json.loads((folder/'result.json').read_text())
        if result['status'] != 'passed':
            raise RuntimeError(f'micro {case} did not pass; stop diagnostics')
        checkpoints.append(result['checkpoint'])
    with (output/'micro-results.json').open('x') as stream:
        json.dump(dict(grounding_checkpoint=args.grounding_checkpoint, checkpoints=checkpoints), stream, indent=2)


if __name__ == '__main__':
    main()
