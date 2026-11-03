import argparse
import json
from pathlib import Path
from .builder import build_dataset
from .splits import audit_dataset


def main():
    parser = argparse.ArgumentParser(description="Phase 1 dataset pilot (no training or robot simulation)")
    sub = parser.add_subparsers(dest="command",required=True)
    build = sub.add_parser("build")
    build.add_argument("--output",type=Path,required=True)
    build.add_argument("--count",type=int,default=100)
    build.add_argument("--seed",type=int,default=1202)
    audit = sub.add_parser("audit")
    audit.add_argument("path",type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build_dataset(args.output,args.count,args.seed)
        path = args.output
    else:
        path = args.path
    result = audit_dataset(path)
    print(json.dumps(result,indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
