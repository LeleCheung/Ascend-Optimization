#!/usr/bin/env python3
"""给归档的 PReLU 候选追加融合误差补偿归约，不覆盖原文件。"""

import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("输出文件已存在")
    here = Path(__file__).resolve().parent
    source = args.input.read_text(encoding="utf-8")
    source += "\n\n" + (here / "prelu-compensated-reduction.py").read_text(encoding="utf-8")
    source += "\n\n" + (here / "prelu-accurate-fused.py").read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(source.encode("utf-8"))
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
