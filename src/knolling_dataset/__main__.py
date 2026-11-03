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
    build.add_argument("--teacher-policy",choices=("legacy_v1_2","balanced_rows_v1.3"),default="balanced_rows_v1.3")
    audit = sub.add_parser("audit")
    audit.add_argument("path",type=Path)
    audit.add_argument("--read-only",action="store_true")
    migration=sub.add_parser("migrate")
    migration.add_argument("--parent",type=Path,required=True)
    migration.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "migrate":
        from .migration import migrate_dataset
        print(json.dumps(migrate_dataset(args.parent,args.output),indent=2))
        return
    if args.command == "build":
        build_dataset(args.output,args.count,args.seed,policy=args.teacher_policy)
        path = args.output
    else:
        path = args.path
    result = audit_dataset(path,read_only=getattr(args,"read_only",False))
    print(json.dumps(result,indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
