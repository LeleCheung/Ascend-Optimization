你是Ascend 910B算子优化工程师。只根据提供的源码和测量，生成一个更快的完整单文件候选。

输出JSON，仅含source（完整Python源码）、hypothesis（中文）、expected_effect（中文）。

不得调用工具、读取文件或联网。保留def run(A)的原地返回别名和完整shape/dtype/stride语义。

KGS要求候选中若存在与算子同名的公共符号，则必须与run是同一函数对象；可只定义run，或在文件末尾写算子名=run。

不能调用目标算子的PyTorch或FlagGems作为新捷径。原代码已有的非连续fallback可以原样保留。允许torch分配、设备上下文和stream接口。

不能删用例、读取评测模式、缓存计算结果、缓存输入Tensor或复用旧输入指针。浮点运算和特殊值必须正确。

缓存编译产物时每次重新绑定本次输入、当前stream及配置。使用当前后端实际接口；不新增CUDA专有选项。

只允许一次候选机会。输出完整有效源码，不要Markdown，不要占位符。

算子：selu_

共享接口知识：
# 两组共享的固定后端接口知识

固定环境：910B4-1、CANN9.0、FlagTree0.6.0+ascend.gitc286cba6、Triton3.5.1。先导入 torch、torch_npu，再导入 Triton，避免当前安装版本的循环导入问题。

以下为本机已核对的 JITFunction.run 提交语义，供两组同等使用。它只描述编译和调用接口，不给任何算子的最佳实现。

1. 正常调用 `compiled = kernel[grid](*args, **kwargs)` 返回已编译对象。若是 future，使用 `.result()`。
2. 本机 `kernel.device_caches[device]` 是 `(kernel_cache, kernel_key_cache, target, backend, binder)`。
3. 绑定前，选项执行 `kwargs['debug'] = kwargs.get('debug', kernel.debug) or knobs.runtime.debug`，然后 `triton.runtime.jit._flagprism.apply_compile_options(kwargs)`；`binder(*args, **kwargs)` 返回 `(bound_args, specialization, options)`。直接查缓存必须与正常 JIT 采用同样的选项，否则可能命中不了真实编译产物。
4. `driver.active.get_current_device()`、`driver.active.get_current_stream(device)` 获取本次设备和当前 stream。
5. `grid(bound_args)` 或固定 tuple 得到实际 grid；不足三维用 1 补齐。
6. `metadata = compiled.launch_metadata(grid, stream, *bound_args.values())`。
7. 提交接口为 `compiled.run(gx, gy, gz, stream, compiled.function, compiled.packed_metadata, metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())`。
8. 所有输入指针和标量应来自本次调用。可以缓存编译对象与配置，不能缓存旧输入 Tensor、旧数据指针或固定 stream。新 dtype、shape、stride、设备、编译选项需要匹配正确的特化。
9. 直接调用是版本受限的优化途径，必须保留未知布局／版本的正常入口。禁止修改共享运行库、关闭正确性检查或缓存结果。

小输入、较大输入、连续与非连续输入都需要正确。grid 不超过 65535；容量受表达式和 dtype 影响，不假定 block 越大越好。不从活跃比例推导独占流水时间或定量 Roofline。


原始执行观测：
{"schema": "hgkda-execution-slice-v1", "source_sha256": "ff49ce74694205c20e6c5793361b8cd47ee7089ac1f434fb6d103c990621af72", "operator": "selu_", "probe_sha256": "9ab0cce3f5c701fafe5d32a02522efd280833b80bfc4b62f996ce465036a92e9", "physical_card": 7, "torch": "2.9.0+cpu", "triton": "3.5.1", "jit_runtime_sha256": "d4b9f4cc94c7a5beeb06d8c9404442cdfa26aabc4a474697a090639deb72d2db", "timing_scope": "预热后批量同步walltime/调用，非纯设备时间；相同样本交替测量", "cases": [{"id": "float16-1024", "shape": [1024], "dtype": "float16", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [97.49008333333335, 88.87841666666667, 88.74675, 87.95508333333333, 89.25341666666667, 87.81591666666667, 86.02674999999999], "direct_us": [43.98925, 41.425000000000004, 41.8975, 40.77925, 41.5625, 40.363416666666666, 48.18841666666666]}, "direct_semantics": "passed", "trace": "candidate-trace.json"}, {"id": "float16-1048576", "shape": [1048576], "dtype": "float16", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [287.73108333333334, 117.51258333333334, 147.866, 103.05175000000001, 108.33091666666667, 96.28091666666667, 95.03841666666666], "direct_us": [89.38425000000001, 87.80675000000001, 74.98091666666666, 67.48258333333334, 47.254999999999995, 45.26083333333333, 47.24916666666667]}, "direct_semantics": "passed", "tile_interventions": [{"block": 512, "grid": [2048], "timings": {"parent": [101.260125, 73.0825, 67.92125, 68.633875, 66.631375], "variant": [94.280125, 88.526375, 86.1775, 93.312625, 85.62875]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 2048, "grid": [512], "timings": {"parent": [98.565, 69.790125, 76.911375, 67.582625, 66.818875], "variant": [66.89375, 65.283875, 76.225, 64.69875, 65.07375]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 4096, "grid": [256], "timings": {"parent": [104.027625, 74.118875, 75.4825, 70.663875, 78.4125], "variant": [76.418875, 72.242625, 69.82625, 69.643875, 79.18125]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}]}, {"id": "float32-1024", "shape": [1024], "dtype": "float32", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [102.08341666666666, 93.67591666666668, 90.51425, 92.07008333333333, 90.68091666666668, 91.88425000000001, 90.37841666666667], "direct_us": [45.89175, 42.27583333333333, 45.366749999999996, 41.962583333333335, 43.66916666666666, 42.04925, 42.76925]}, "direct_semantics": "passed"}, {"id": "float32-1048576", "shape": [1048576], "dtype": "float32", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [455.39300000000003, 113.80758333333334, 111.30425000000001, 110.00266666666666, 107.79091666666666, 110.00349999999999, 113.76091666666667], "direct_us": [58.38675, 52.531749999999995, 58.44925, 49.73925, 50.616749999999996, 48.88008333333334, 51.24341666666667]}, "direct_semantics": "passed", "tile_interventions": [{"block": 512, "grid": [2048], "timings": {"parent": [698.7295, 73.27625, 74.097625, 73.592625, 70.570125], "variant": [107.612625, 89.551375, 95.880125, 89.285125, 96.578875]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 2048, "grid": [512], "timings": {"parent": [183.077625, 70.947625, 71.22875, 71.461375, 72.853875], "variant": [69.865125, 68.915125, 70.03375, 68.19125, 67.267625]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 4096, "grid": [256], "timings": {"parent": [317.477875, 79.258875, 82.9925, 76.7125, 80.44125], "variant": [99.932625, 117.230125, 76.6275, 76.18125, 74.81125]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}]}]}

根据以上原始数据自行分析并选择改法；没有额外的机制图或自动诊断建议。

固定FlagGems起点：
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated by KernelGen: https://github.com/flagos-ai/KernelGen
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def selu_kernel_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
    y = y_f32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def selu_(*args, **kwargs):
    logger.debug("GEMS SELU_")
    x = None
    if len(args) > 0 and torch.is_tensor(args[0]):
        x = args[0]
    elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
        x = kwargs["input"]
    elif "self" in kwargs and torch.is_tensor(kwargs["self"]):
        x = kwargs["self"]
    elif "x" in kwargs and torch.is_tensor(kwargs["x"]):
        x = kwargs["x"]
    else:
        raise ValueError(
            "selu_ expects a Tensor as the first argument or under 'input'/'self'/'x' keyword."
        )

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not x.is_contiguous()) or (x.dtype not in supported_dtypes):
        torch.ops.aten.selu_(x)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        selu_kernel_[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x

_hg_layout_parent = selu_
def run(A):
    if not A.is_contiguous():
        temporary = A.contiguous()
        _hg_layout_parent(temporary)
        A.copy_(temporary)
        return A
    return _hg_layout_parent(A)
selu_ = run
