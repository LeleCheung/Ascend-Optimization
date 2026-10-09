#!/usr/bin/env python3
"""从已归档评测合同和修复源码准备独立复测输入。"""

import argparse
import json
from pathlib import Path


OPERATORS = ("_prelu_kernel_backward", "batch_norm_backward", "smooth_l1_loss_backward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {}
    for operator in OPERATORS:
        historical = args.historical_report / "measurements" / operator
        if operator.startswith("_"):
            historical = args.historical_report / "measurements/prelu-retry" / operator
        output = args.output / operator
        output.mkdir()
        request = json.loads((historical / "candidate-1.request.json").read_text(encoding="utf-8"))
        result = json.loads((historical / "candidate-1.result.json").read_text(encoding="utf-8"))
        if result["status"] != "PASSED":
            raise ValueError(f"{operator} 历史候选未通过完整评测")
        blueprint = {key: request[key] for key in ("binding", "settings")}
        (output / "blueprint.json").write_text(json.dumps(blueprint, indent=2) + "\n", encoding="utf-8")
        for src, filename in ((historical / "inspect.json", "historical-inspect.json"),
                              (historical / "candidate.py", "candidate.py"),
                              (args.repairs / operator / "repaired.py", "repaired.py"),
                              (args.repairs / operator / "correctness.patch", "correctness.patch")):
            (output / filename).write_bytes(src.read_bytes())
        if "bundle_id" in blueprint["binding"]:
            (output / "operator-bundle.tar").write_bytes((historical / "operator-bundle.tar").read_bytes())
        protocol[operator] = {"workloads": {w["uuid"]: {"phase": w["phase"], "axes": w["axes"]}
                                            for w in result["per_workload"]},
                              "historical_candidate": "2026-10-09 archived unified comparison"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
