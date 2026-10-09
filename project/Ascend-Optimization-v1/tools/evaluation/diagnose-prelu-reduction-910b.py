#!/usr/bin/env python3
"""比较原修复版、FP32 中间缓冲及原生接口的权重梯度归约精度。"""

import importlib.util
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext

import torch
import torch_npu
import flag_gems


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    original = module("prelu_repair_v1", "repaired.py")
    fp32 = module("prelu_repair_fp32", "repaired-fp32.py")
    fp32sum = module("prelu_repair_fp32sum", "repaired-fp32sum.py")
    candidate = module("prelu_candidate", "candidate.py")
    rows = []
    shape = (16, 7, 57, 32, 29)
    for dtype in (torch.bfloat16, torch.float16):
        for seed in range(8):
            torch.manual_seed(seed)
            grad = torch.randn(shape, dtype=dtype, device="npu:0")
            x = torch.randn(shape, dtype=dtype, device="npu:0")
            weight = torch.full((1,), 0.25, dtype=dtype, device="npu:0")
            reference = torch.ops.aten._prelu_kernel_backward(grad, x, weight)[1]
            outputs = {}
            for context_name, context in (("native", nullcontext()), ("gems", flag_gems.use_gems())):
                with context:
                    outputs.update({
                        context_name + "_repair_v1": original.run(grad, x, weight)[1],
                        context_name + "_repair_fp32": fp32.run(grad, x, weight)[1],
                        context_name + "_repair_fp32sum": fp32sum.run(grad, x, weight)[1],
                        context_name + "_candidate": candidate.run(grad, x, weight)[1],
                        context_name + "_torch_low_product_sum": torch.where(x < 0, grad * x, 0.0).sum().reshape(weight.shape),
                        context_name + "_torch_fp32_product_sum": torch.where(x < 0, grad.float() * x.float(), 0.0).sum().to(dtype).reshape(weight.shape),
                    })
            torch.npu.synchronize()
            row = {"dtype": str(dtype), "seed": seed, "shape": list(shape), "reference": reference.item()}
            for name, output in outputs.items():
                difference = abs(output.item() - reference.item())
                # 历史测试要求 BF16 1.6% 相对容差，FP16 0.1%；不放宽。
                tolerance = (0.016 if dtype == torch.bfloat16 else 0.001) * abs(reference.item()) + 1e-4
                row[name] = {"value": output.item(), "abs_difference": difference, "passed": difference <= tolerance}
            rows.append(row)
            print(json.dumps(row), flush=True)
    destination = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
    (destination / "precision-results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
