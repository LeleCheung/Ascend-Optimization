#!/usr/bin/env python
"""Run auto_gen example.

Usage:
    python kgrunner/examples/auto_gen/run_example.py --input ops.jsonl
    python kgrunner/examples/auto_gen/run_example.py --input ops.jsonl --gpu-ids 0 1

Input JSONL format (one JSON object per line):
    {"OPERATOR": "relu"}
    {"OPERATOR": "gelu"}
    {"OPERATOR": "silu"}
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from kgrunner import run_parallel, load_agent, GPUPool

AGENT_DIR = Path(__file__).resolve().parent
agent_def = load_agent(AGENT_DIR)


def main():
    parser = argparse.ArgumentParser(description="auto_gen: generate FlagGems operators via kgrunner")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file, one task per line")
    parser.add_argument("--gpu-ids", type=int, nargs="*", help="GPU device IDs, e.g. --gpu-ids 0 1 2 3 (default: auto-detect all)")
    args = parser.parse_args()

    pool = GPUPool(device_ids=args.gpu_ids)

    results = run_parallel(
        agent_def, args.input,
        gpu=pool,
    )

    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
