---
schema_version: '1.0'
id: kg:diagnostic:triton-ascend-precision-operation-isolation
kind: diagnostic
title: Isolate Triton Ascend precision failures by atomic operation and lowering path
summary: Reduce an end-to-end precision failure to the first divergent operation,
  then compare scalar and vector lowering paths before applying and validating a fix.
claim_key: diagnostic.triton_ascend.precision_operation_isolation
domains:
- compiler
- diagnosis
- numerics
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:58.566728Z'
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:53:50.775107Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/triton-precision-debug/references/precision-alignment-guide.md#排查全流程：五阶隔离法
  title: Isolate Triton Ascend precision failures by atomic operation and lowering
    path
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  numerics:
    exact: false
retrieval:
  phases:
  - post_error
  - post_evaluation
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - precision_failure
  - ulp_difference
  - quantization_mismatch
  techniques:
  - operation_isolation
  - microbenchmark
  - scalar_vector_comparison
  keywords:
  - precision
  - isolation
  - microbenchmark
  - scalar
  - vector
  - ULP
evidence_state: source_supported
managed:
  content_hash: sha256:b6cd4607c1311f948442fb50d6430e595a6c88f965812255afb9d33babed3c8d
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:58.566728Z'
  updated_at: '2026-07-27T07:53:50.775107Z'
---

# Claim

Reduce an end-to-end precision failure to the first divergent operation, then compare scalar and vector lowering paths before applying and validating a fix.

# Source material

## 排查全流程：五阶隔离法

### Stage 0：前置检查（排除低级错误）

在怀疑编译器之前，先确认以下基础项：

```
□ 输出 shape 与参考完全一致
□ NaN 位置完全一致（mask 按位相等）
□ Inf 位置和符号完全一致
□ dtype 转换逻辑正确（如 bfloat16 -> float32 后再计算）
□ 未使用未初始化的 memory（tl.load 的 other=0.0 是否合适）
```

### Stage 1：端到端差异定位

先用最小 reproduction 找到差异最大的 case，缩小排查范围。

```python
# 在失败 case 上对比 Triton 输出与 torch 输出
triton_out, triton_scale = model_new(x, smooth)
torch_out, torch_scale = model_ref(x, smooth)

diff = (triton_out.float() - torch_out.float()).abs()
print(f"max_diff={diff.max().item():.2e}")
print(f"diff>0 count={(diff > 0).sum().item()}")

# 如果 diff 存在，打印前几处差异的原始值和量化值
indices = torch.where(diff > 0)
for i in range(min(10, len(indices[0]))):
    idx = tuple(t[i].item() for t in indices)
    print(f"idx={idx}: triton={triton_out[idx].item()}, torch={torch_out[idx].item()}")
```

**判定**：若 diff 存在且原始输入值差异为 0（即差异纯由量化引入），则进入 Stage 2。

### Stage 2：逐操作隔离微基准测试

将算子拆分为独立原子操作，每个操作写一个最小 Triton kernel，与 torch 一对一比对。

**检查清单（DynamicQuant 为例）**：

| # | 操作 | Triton 代码 | Torch 参考 | 预期结果 |
|---|------|------------|-----------|---------|
| 1 | `abs` | `tl.abs(x)` | `x.abs()` | 0 diff |
| 2 | `max` | `tl.max(abs_x, axis=0)` | `x.abs().max(dim=1)[0]` | 0 diff |
| 3 | 常量除法 | `max_val / 127.0` | `max_val / 127.0` | **需重点检查** |
| 4 | clamp scale | `tl.maximum(scale, 1e-10)` | `scale.clamp(min=1e-10)` | 0 diff |
| 5 | 变量除法 | `x / scale` | `x / scale` | 0 diff |
| 6 | round | `nearbyint(x)` | `torch.round(x)` | 0 diff |
| 7 | clamp quant | `max(min(val,127),-128)` | `val.clamp(-128,127)` | 0 diff |
| 8 | cast to int8 | `.to(tl.int8)` | `.to(torch.int8)` | 0 diff |

**微基准 kernel 模板（以 scale 除法为例）**：

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def scale_kernel(max_ptr, scale_ptr, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    max_vals = tl.load(max_ptr + offsets, mask=mask, other=0.0)
    scales = max_vals / 127.0
    scales = tl.maximum(scales, 1e-10)
    tl.store(scale_ptr + offsets, scales, mask=mask)

# 测试数据
torch.manual_seed(0)
x = torch.randn((512, 512), dtype=torch.bfloat16, device="npu")
max_abs = x.float().abs().max(dim=1)[0]

# Torch 参考
scale_torch = max_abs / 127.0
scale_torch = scale_torch.clamp(min=1e-10)

# Triton 测试
scale_triton = torch.empty_like(max_abs)
grid = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE,)
scale_kernel[grid](max_abs, scale_triton, M, BLOCK_SIZE=BLOCK_SIZE)
torch.npu.synchronize()

# 比对
diff = (scale_torch.cpu() - scale_triton.cpu()).abs()
print(f"BLOCK_SIZE={BLOCK_SIZE}: max_diff={diff.max().item():.2e}, count={(diff > 0).sum().item()}")
```

### Stage 3：Scalar vs Vector 路径判定

当 Stage 2 发现某个操作存在差异时，用以下矩阵测试确定是否为 scalar/vector 编译器行为差异：

| 测试 | BLOCK_SIZE | Grid | 预期 |
|------|-----------|------|------|
| A | 1 | (M,) | **可能失败**（scalar 优化） |
| B | 2 | (M/2,) | 应通过（vector） |
| C | 4 | (M/4,) | 应通过（vector） |
| D | 2048 | (1,) | 应通过（vector loop） |
| E | scalar loop in kernel | (1,) | **可能失败**（显式 scalar） |

**测试 E 的代码形式**：
```python
@triton.jit
def scalar_loop_kernel(max_ptr, scale_ptr, M):
    for i in range(M):
        val = tl.load(max_ptr + i)
        scale = val / 127.0          # 显式标量除法
        tl.store(scale_ptr + i, scale)
```

**判定规则**：
- 若 A/E 失败但 B/C/D 通过 → **确诊 scalar/vector 编译器差异**
- 若全部失败 → 差异来源不是 scalar/vector，需继续排查（如 `127.0` 字面量精度、div 指令选型等）

### Stage 4：验证修复方案

确诊后，将算子中对应操作改为 vector 路径，然后做**两级验证**：

**Level 1 — 单操作验证**：仅修改目标操作，其余仍用 torch，确认该操作单独通过。

**Level 2 — 全链路验证**：将修改后的操作放回完整 pipeline，跑全量 case 验证。

# Applicability

The structured scope on this Concept is normative.
