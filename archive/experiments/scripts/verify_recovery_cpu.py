#!/usr/bin/env python3
"""Run review checks only; deliberately has no train/device/update arguments."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys


def main():
    if Path(sys.prefix).name != 'knolling':
        raise RuntimeError('run this checker with conda run -n knolling python')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-model', action='store_true',
                        help='also check cached R18/MiniLM on CPU; never downloads weights')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checked = []
    for directory in ('src', 'tests', 'scripts'):
        for path in (root/directory).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            checked.append(path)
    print(f'Python syntax: {len(checked)} files passed', flush=True)
    subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', RUN_PRETRAINED_TESTS='0',
               RUN_RECOVERY_CUDA_TESTS='0',
               RUN_RECOVERY_PRETRAINED_TESTS='1' if args.cached_model else '0')
    # Unit tests use temporary directories; legacy artifact locations are read-only inputs.
    subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-ra'], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
