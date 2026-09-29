# 灰名单持久化指南

## 概述

`log_graylist_hits()` 函数支持将灰名单命中记录持久化到磁盘，便于后续人工审查。

**默认行为**：自动持久化到 `kernelgen_server/evaluation/graylists/log/` 目录。

**重要变更（2026-08-11）**：
- ✅ `.to()` 方法已从灰名单中移除（它是 dtype/device 转换，返回 copy 而非 view）
- ✅ 持久化默认启用，保存到 `graylists/log/` 目录

## 使用方式

### 1. 在代码中调用

```python
from kernelgen_server.evaluation.hack_detection import detect_obvious_hack, log_graylist_hits

# 黑名单检测（硬信号）
hack = detect_obvious_hack(implementation)

# 灰名单检测（软信号，写日志 + 默认持久化到 graylists/log/）
log_graylist_hits(
    implementation,
    vendor="muxi",                      # 厂商名（用于文件名前缀）
    # persist_dir 默认 "default" → 自动使用 graylists/log/
    # persist_dir=None → 禁用持久化，仅记录日志
    # persist_dir="/custom/path" → 使用自定义路径
)
```

### 2. 使用扫描脚本批量处理

```bash
# 扫描所有算子并持久化灰名单（使用自定义路径）
python tests/scan_operators.py \
    --persist-graylist /data/bwzhou/kernel_operators_graylist \
    --json /data/bwzhou/scan_results.json

# 只扫描某个厂商
python tests/scan_operators.py \
    --vendor muxi \
    --persist-graylist /tmp/muxi_graylist

# 只打印，不持久化
python tests/scan_operators.py --vendor muxi
```

## 持久化格式

每个有灰名单命中的算子生成一个 JSON 文件，命名格式：`{vendor}_{operator}.json`

### 示例：`muxi_conj.json`

```json
{
  "operator": "conj",
  "vendor": "muxi",
  "language": "triton",
  "hit_count": 3,
  "gray_calls": [],
  "gray_methods": [
    {
      "file": "conj.py",
      "line": 21,
      "method": "numel"
    },
    {
      "file": "conj.py",
      "line": 22,
      "method": "view"
    },
    {
      "file": "conj.py",
      "line": 27,
      "method": "view"
    }
  ]
}
```

**注意**：示例中不再包含 `.to()` 方法（已从灰名单中移除）。

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `operator` | string | 算子名（不含厂商前缀） |
| `vendor` | string | 厂商名 |
| `language` | string | 实现语言（通常为 "triton"） |
| `hit_count` | int | 灰名单命中总数 |
| `gray_calls` | array | 命名空间形式调用（如 `torch.clone`） |
| `gray_methods` | array | 方法形式调用（如 `.reshape`、`.view`） |

每个命中项包含：
- `file`: 源文件名
- `line`: 行号
- `call` / `method`: 调用的 API 名称

## 日志输出

无论是否持久化，灰名单命中都会写入日志：

```
WARNING ...hack_detection: GRAYLIST [muxi/conj] 3 hit(s): graylist Tensor method: conj.py:21 .numel; graylist Tensor method: conj.py:22 .view; ...
```

干净的实现（无命中）输出 DEBUG 级别日志：

```
DEBUG ...hack_detection: GRAYLIST [muxi/pure_kernel] clean (no view/layout ops detected)
```

## 典型工作流

### 1. 批量扫描并持久化

```bash
python tests/scan_operators.py \
    --persist-graylist /data/bwzhou/kernel_operators_graylist \
    --json /data/bwzhou/scan_results.json
```

生成：
- 356 个 JSON 文件（每个有灰名单命中的算子一个）
- 1 个汇总 JSON（`scan_results.json`）

### 2. 人工审查

查看特定算子的灰名单命中：

```bash
cat /data/bwzhou/kernel_operators_graylist/muxi_conj.json | jq
```

按命中次数排序（找出高频使用 plumbing 的算子）：

```bash
cd /data/bwzhou/kernel_operators_graylist
for f in *.json; do
    count=$(jq '.hit_count' "$f")
    echo "$count $f"
done | sort -rn | head -20
```

### 3. 统计分析

```python
import json
from pathlib import Path
from collections import Counter

graylist_dir = Path("/data/bwzhou/kernel_operators_graylist")
method_usage = Counter()

for path in graylist_dir.glob("*.json"):
    record = json.loads(path.read_text())
    for item in record["gray_methods"]:
        method_usage[item["method"]] += 1

print("最常用的 Tensor 方法:")
for method, count in method_usage.most_common(10):
    print(f"  .{method}: {count} 次")
```

## 当前扫描结果（363 个算子）

| Vendor | Total | ❌ Hack | ⚠ Gray | ✅ Clean |
|--------|-------|---------|--------|----------|
| haiguang | 45 | 0 | 43 | 2 |
| huawei | 134 | 0 | 131 | 3 |
| kunlunxin | 15 | 0 | 15 | 0 |
| moer | 48 | 0 | 48 | 0 |
| muxi | 47 | 0 | 47 | 0 |
| pingtouge | 36 | 0 | 35 | 1 |
| tianshu | 38 | 0 | 37 | 1 |
| **TOTAL** | **363** | **0** | **356** | **7** |

**关键发现：**
- 0 个黑名单命中（所有实现都是合法 Triton）
- 356 个灰名单命中（使用了视图/布局 plumbing）
- 7 个完全干净（纯 Triton，无任何 PyTorch API 调用）

高频灰名单操作（`.to()` 已从灰名单中移除）：
- `.stride` — 获取内存布局信息
- `.contiguous` — 强制连续内存布局
- `.view` / `.reshape` — 视图变换
- `.numel` — 元素总数查询
- `.dim` — 获取维度数

**已排除的操作**（不属于灰名单）：
- `.to()` — dtype/device 转换（返回 copy，不是 view）
- `torch.zeros`, `torch.empty` 等 — 张量创建（应使用 Triton 原生 API）

## 与黑名单检测的关系

| 检测类型 | 函数 | 输出 | 语义 |
|----------|------|------|------|
| 黑名单 | `detect_obvious_hack()` | `HackDetection(is_hack, reason)` | **硬信号**：违规 fallback |
| 灰名单 | `log_graylist_hits()` | 日志 + 默认持久化到 `graylists/log/` | **软信号**：合法 plumbing，仅供参考 |

两者可同时使用：

```python
hack = detect_obvious_hack(impl)
log_graylist_hits(impl, vendor="muxi", persist_dir="./graylist")

if hack.is_hack:
    print("❌ 违规:", hack.hack_reason)
else:
    print("✓ 合法实现（可能使用了 plumbing，见灰名单日志/文件）")
```

## 另见

- [GRAYLIST_USAGE.md](GRAYLIST_USAGE.md) — 灰名单检测使用指南
- [hack_detection.py](hack_detection.py) — 黑名单检测实现
- [graylist_detection.py](graylist_detection.py) — 灰名单检测实现
- [tests/scan_operators.py](../../tests/scan_operators.py) — 批量扫描脚本
