"""Explicit S0 preparation/training commands. No action defaults to training."""
import argparse
import json

from .runner import read_config, preflight, train


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'train'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', default='configs/s0.json')
        sub.add_argument('--device', default='cpu' if name == 'preflight' else 'cuda')
        sub.add_argument('--download-weights', action='store_true', help='allow downloading pinned public pretrained weights')
        if name == 'preflight':
            sub.add_argument('--report', default='artifacts/s0_verification/preflight.json')
        else:
            sub.add_argument('--run-dir', required=True, help='new output directory, also for resume')
            sub.add_argument('--resume', help='checkpoint to continue into a new output directory')
            sub.add_argument('--start-training', action='store_true', required=True,
                             help='explicitly enable optimizer updates')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    config = read_config(args.config)
    if args.command == 'preflight':
        result = preflight(config, args.report, args.device, not args.download_weights)
        print(json.dumps({k:v for k,v in result.items() if k != 'provenance'}, indent=2))
    else:
        result = train(config, args.run_dir, args.device, not args.download_weights, args.resume)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
