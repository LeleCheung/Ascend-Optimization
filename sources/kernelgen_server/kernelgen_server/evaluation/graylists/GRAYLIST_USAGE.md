# 灰名单检测使用指南

## 概述

`graylist_detection.py` 是 `hack_detection.py` 的补充模块，用于检测 Triton 实现中使用了哪些**模糊地带的视图/布局操作**（如 reshape、transpose、stride、copy 等）。

**重要**：灰名单**不包含**张量创建操作（zeros、ones、empty、rand 等）——这些明确的分配调用应该由 Triton 自己处理（tl.zeros、tl.load），而非 torch API。

灰名单命中是**弱信号、仅供参考**，不应用于拒绝或惩罚提交。它的作用是提供审查上下文：了解一个实现依赖了哪些视图/布局操作。

## 快速开始

```python
from kernelgen_server.evaluation.graylist_detection import detect_graylist_usage
from kernelgen_server.protocol.schema import Implementation

result = detect_graylist_usage(implementation)

if result.has_hits():
    print("该实现使用了以下视图/布局操作:")
    print(result.summary())
else:
    print("该实现未使用任何模糊的视图/布局操作")
```

## 返回结构

`GraylistDetection` 数据类包含：

- `gray_calls: tuple[tuple[str, int, str], ...]`  
  命名空间形式的调用，每项为 `(文件路径, 行号, 规范化名称)`  
  例如：`("main.py", 10, "torch.empty_like")`

- `gray_methods: tuple[tuple[str, int, str], ...]`  
  方法形式的调用，每项为 `(文件路径, 行号, 方法名)`  
  例如：`("main.py", 12, "reshape")`

- `has_hits() -> bool`  
  是否有任何灰名单命中

- `summary() -> str`  
  人类可读的摘要，格式：  
  `graylist Torch API: main.py:10 torch.empty_like; graylist Tensor method: main.py:12 .reshape`

## 与黑名单检测的关系

| 模块 | 检测目标 | 列表来源 | 命中语义 |
|------|---------|---------|---------|
| `hack_detection.py` | 明确禁止的 compute 算子 | `blacklists/*.yaml` 中的 `forbidden:` | **硬信号**：`is_hack = True` |
| `graylist_detection.py` | 模糊的视图/布局操作 | `graylists/*.yaml` 中的 `gray:` | **软信号**：仅记录，不影响合法性 |

数学关系：

```
graylist(namespace) = public_callables(namespace) - blacklist.forbidden(namespace) - tensor_creation_ops
```

对于 `torch` 命名空间（torch 2.12.0）：
- 黑名单（forbidden）：538 个禁止的 compute 算子
- 张量创建（excluded）：40 个分配操作（zeros、ones、empty、rand 等）
- 灰名单（gray）：151 个模糊的视图/布局操作
- 总和：729 个公开可调用

**关键区别**：
- 张量创建（`torch.zeros`、`torch.empty_like`）→ 不在灰名单中，Triton 应该用自己的分配机制
- 视图/布局（`torch.reshape`、`torch.transpose`、`.contiguous()`）→ 在灰名单中，这是模糊地带

## 覆盖的调用形式

### 命名空间形式

别名会被解析到规范名称：

```python
import torch
torch.clone(x)                    # → torch.clone

import torch as tr
tr.transpose(x, 0, 1)             # → torch.transpose

from torch import reshape as r
r(x, (-1,))                       # → torch.reshape
```

### 方法形式

Tensor 方法按裸属性名匹配（接收者类型静态未知）：

```python
x.reshape(-1)      # → .reshape（命中灰名单）
x.contiguous()     # → .contiguous（命中灰名单）
x.numel()          # → .numel（命中灰名单）
```

**重要**：Triton 命名空间调用（`tl.load`、`tl.store`、`libdevice.tanh`）会被正确识别并排除，不会误判为 Tensor 方法。

## 典型应用场景

### 1. 审查上下文

```python
result = detect_graylist_usage(impl)
if result.has_hits():
    print(f"提示：该实现依赖了 {len(result.gray_calls) + len(result.gray_methods)} 处视图/布局操作")
    print(result.summary())
```

### 2. 统计分析

```python
from collections import Counter

view_usage = Counter()
for impl in all_implementations:
    result = detect_graylist_usage(impl)
    for _, _, name in result.gray_calls:
        view_usage[name] += 1
    for _, _, method in result.gray_methods:
        view_usage[f".{method}"] += 1

print("最常用的视图/布局操作:")
for name, count in view_usage.most_common(10):
    print(f"  {name}: {count} 次")
```

### 3. 与黑名单检测组合

```python
from kernelgen_server.evaluation.hack_detection import detect_obvious_hack
from kernelgen_server.evaluation.graylist_detection import detect_graylist_usage

hack_result = detect_obvious_hack(impl)
gray_result = detect_graylist_usage(impl)

if hack_result.is_hack:
    print("❌ 检测到明确违规的 Torch fallback")
    print(hack_result.hack_reason)
elif gray_result.has_hits():
    print("⚠ 合法实现，但使用了以下视图/布局操作:")
    print(gray_result.summary())
else:
    print("✓ 纯 Triton 实现，无任何模糊操作")
```

## 限制与边界情况

1. **方法形式是最佳努力**  
   无法在静态分析中确定接收者类型，因此 `custom_obj.reshape(...)` 也会命中（即使 `custom_obj` 不是 Tensor）。

2. **动态调用无法覆盖**  
   `getattr(x, "reshape")(-1)` 不会被检测到。

3. **语法错误文件被跳过**  
   解析失败的源文件会被静默跳过（实现加载器负责语法错误的报告）。

4. **只检测 Triton 实现**  
   `language != "triton"` 的实现直接返回空结果。

5. **张量创建不在灰名单中**  
   `torch.zeros`、`torch.empty_like`、`torch.rand` 等明确的分配操作已从灰名单中排除。这些应该由 Triton 自己处理（`tl.zeros`、预分配输出）。

## 维护

灰名单与黑名单一样会随 PyTorch 版本漂移。升级 torch 后：

```bash
cd kernelgen_server/evaluation/graylists
python _generate.py > /tmp/new_graylists.txt
# 检查差异，更新 yaml 文件
```

详见 `graylists/README.md`。

## 另见

- `hack_detection.py` — 黑名单检测（硬信号）
- `blacklists/README.md` — 黑名单维护指南
- `graylists/README.md` — 灰名单生成与维护
