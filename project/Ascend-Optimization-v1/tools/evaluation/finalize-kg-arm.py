#!/usr/bin/env python3
"""导出已结束 KG 组的最佳候选，再经 KGS 独立完整复验。"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("operator")
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:19655")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    exported = args.output / "export"
    # 导出工具会检查终态、算子身份、完整通过及源码快照。
    subprocess.run([
        sys.executable, str(tools / "export-kg-best.py"),
        str(args.workspace), args.operator, str(exported),
    ], check=True)
    subprocess.run([
        sys.executable, str(tools / "evaluate-operator.py"),
        "--operator", args.operator, "--source", str(exported / "main.py"),
        "--output", str(args.output / "independent"), "--label", args.label,
        "--server", args.server, "--timeout", str(args.timeout),
    ], check=True)
    print("KG_INDEPENDENT_VALIDATION_PASSED", args.operator, args.label, flush=True)


if __name__ == "__main__":
    main()
