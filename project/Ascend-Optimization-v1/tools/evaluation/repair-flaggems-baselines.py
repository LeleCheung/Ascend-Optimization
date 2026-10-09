#!/usr/bin/env python3
"""从历史源码生成最小正确性修复版，不改动 FlagGems checkout 或原始归档。"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path


OPERATORS = ("_prelu_kernel_backward", "batch_norm_backward", "smooth_l1_loss_backward")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"修复锚点数量不为 1：{old!r}")
    return source.replace(old, new, 1)


def repair_prelu(source, reduction="compensated"):
    source = replace_once(source, "    C,\n    BLOCK_SIZE: tl.constexpr,", "    C,\n    S,\n    BLOCK_SIZE: tl.constexpr,")
    source = replace_once(source, "    c = offsets % C", "    c = (offsets // S) % C")
    source = replace_once(source, "        return grad_input, grad_weight\n", "        return grad_input, torch.zeros_like(weight)\n")
    source = replace_once(source, "        C = x.shape[-1]", "        C = x.shape[1] if ndim >= 2 else 1")
    source = source.replace("last dimension size", "channel dimension size")
    source = replace_once(source, "    C = max(int(C), 1)", "    C = max(int(C), 1)\n    S = x.numel() // (x.shape[0] * C) if ndim >= 2 and C > 1 else 1")
    source = replace_once(source, "        C,\n        BLOCK_SIZE=BLOCK_SIZE,", "        C,\n        S,\n        BLOCK_SIZE=BLOCK_SIZE,")
    if reduction == "compensated":
        scalar_reduction = "    # 保留输入 dtype 的乘积舍入；低精度输入使用 FP32 归约。\n    if weight.numel() == 1 and x.dtype == torch.float32:\n        grad_weight = _prelu_stable_scalar_sum(grad_weight, weight)\n    elif weight.numel() == 1:\n        grad_weight = grad_weight.sum(dtype=torch.float32).reshape(weight.shape)\n"
    else:
        scalar_reduction = "    # 保留输入 dtype 的乘积舍入；标量权重使用与候选一致的 FP32 分阶段归约。\n    if weight.numel() == 1:\n        grad_weight = _prelu_stable_scalar_sum(grad_weight, weight)\n"
    source = replace_once(source, "    return grad_input, grad_weight\n", scalar_reduction + "    else:\n        grad_weight = grad_weight.reshape(x.shape[0], C, S).sum(dim=(0, 2), dtype=torch.float32).reshape(weight.shape)\n    return grad_input, grad_weight.to(dtype=weight.dtype)\n")
    # 当前 torch_npu 接口返回归约后的权重梯度，保留原逐元素内核，补充实际归约。
    source = replace_once(source,
        "    # Weight index is based on the last dimension\n    # For 1D input: c = offset % C\n    # For 2D input (M, N): c = offset % N (column index)\n    # For 3D input (M, N, P): c = offset % P (last dimension index)",
        "    # 权重按通道维（dim=1）索引；标量权重统一使用第 0 项。")
    helper = "prelu-compensated-reduction.py" if reduction == "compensated" else "prelu-weight-reduction.py"
    source += "\n\n" + (Path(__file__).parent / helper).read_text(encoding="utf-8")
    source += "\n\n_flaggems_repaired_prelu = _prelu_kernel_backward\ndef run(grad_output, self, weight):\n    return _flaggems_repaired_prelu(grad_output, self, weight)\n_prelu_kernel_backward = run\n"
    return source


def repair_batch_norm(source):
    source = replace_once(source, "    bias_grad_mask: tl.constexpr,\n    BLOCK_M:",
                          "    bias_grad_mask: tl.constexpr,\n    EPS: tl.constexpr,\n    TRAIN: tl.constexpr,\n    BLOCK_M:")
    source = replace_once(source, "    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)",
                          "    # 当前 torch_npu 保存统计量参数实际为方差。\n    variance = tl.load(feat_pid + inv_std_pointer).to(tl.float32)\n    inv_std = rsqrt(variance + EPS)")
    source = replace_once(source,
        "            curr_input_grad = (\n                inv_std\n                * weight\n                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)\n            )",
        "            if TRAIN:\n                curr_input_grad = (\n                    inv_std * weight\n                    * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)\n                )\n            else:\n                curr_input_grad = inv_std * weight * curr_output_grad")
    source = replace_once(source, "    input_3d = make_3d_for_bn(input)\n    output_grad_3d",
                          "    if output_mask is None:\n        output_mask = (True, True, True)\n    input_3d = make_3d_for_bn(input)\n    output_grad_3d")
    source = replace_once(source, "            save_mean,\n            save_invstd,",
                          "            save_mean if train else running_mean,\n            save_invstd if train else running_var,")
    source = replace_once(source, "            *input_grad.stride(),\n            *output_mask,",
                          "            *(input_grad.stride() if input_grad is not None else input_3d.stride()),\n            *output_mask,\n            EPS=float(eps),\n            TRAIN=bool(train),")
    source = replace_once(source, "        input_grad.view_as(input),", "        input_grad.view_as(input) if input_grad is not None else None,")
    source += "\n\nrun = batch_norm_backward\n"
    return source


def repair_smooth_l1(source):
    source = replace_once(source, 'grad = tl.where(diff == 0.0, float("nan"), tl.where(diff > 0.0, 1.0, -1.0))',
                          'grad = tl.where(diff == 0.0, 0.0, tl.where(diff > 0.0, 1.0, -1.0))')
    source += "\n\nrun = smooth_l1_loss_backward\n"
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prelu-reduction", choices=("compensated", "staged"), default="compensated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    changes = []
    for operator, repair in zip(OPERATORS, (lambda s: repair_prelu(s, args.prelu_reduction), repair_batch_norm, repair_smooth_l1)):
        original = (args.historical_report / "source-audit" / operator / "flaggems-original-source.py").read_bytes()
        fixed = repair(original.decode("utf-8"))
        folder = args.output / operator
        folder.mkdir()
        (folder / "original.py").write_bytes(original)
        (folder / "repaired.py").write_bytes(fixed.encode("utf-8"))
        patch = difflib.unified_diff(original.decode("utf-8").splitlines(True), fixed.splitlines(True),
                                     fromfile="original.py", tofile="repaired.py")
        (folder / "correctness.patch").write_text("".join(patch), encoding="utf-8")
        changes.append({"operator": operator, "original_sha256": hashlib.sha256(original).hexdigest(),
                        "repaired_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
                        "baseline": "历史 FlagGems 原版的本机最小正确性修复；不是未修改的上游 master"})
    (args.output / "repairs.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(changes, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
