#!/usr/bin/env python3
"""保留首批失败证据，将完成的两批复测汇入最终结果目录。"""

import argparse
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--followup", default="followup-prelu-accurate")
    parser.add_argument("--final", default="validated-v5")
    args = parser.parse_args()
    root = args.report
    followup = root / args.followup
    final = root / args.final
    batches = {"_prelu_kernel_backward": followup,
               "batch_norm_backward": root, "smooth_l1_loss_backward": root}
    experiments = []
    for batch in (root, followup):
        if (batch / "exit-code.txt").read_text().strip() != "0":
            raise ValueError(f"批次尚未正常结束：{batch}")
        experiments.append(read(batch / "measurements/experiment.json"))
    if any(e["repeats"] != 3 for e in experiments):
        raise ValueError("每个批次必须设置三轮")
    if len({str(e["physical_device"]) for e in experiments}) != 1:
        raise ValueError("两批使用的物理设备不同")
    for operator, batch in batches.items():
        source = batch / "measurements" / operator
        if (source / "repaired.py").read_bytes() != (final / "repairs" / operator / "repaired.py").read_bytes():
            raise ValueError(f"修复源码不一致：{operator}")
        for repeat in range(1, 4):
            for label in ("repaired-flaggems", "candidate"):
                result = read(source / f"{label}-{repeat}.result.json")
                if result["status"] != "PASSED":
                    raise ValueError(f"未通过：{operator}/{label}/{repeat}")
    measurements = final / "measurements"
    measurements.mkdir(exist_ok=False)
    experiment = {**experiments[0], "operator_batches": {
        op: str(batch.relative_to(root)) for op, batch in batches.items()},
        "batch_experiments": experiments}
    (measurements / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(root / "protocol.json", final / "protocol.json")
    for operator, batch in batches.items():
        shutil.copytree(batch / "measurements" / operator, measurements / operator)
    print(final)


if __name__ == "__main__":
    main()
