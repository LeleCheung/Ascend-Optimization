---
name: kernel-optimize
description: "Use this agent for exactly one measured optimization direction on an existing FlagGems kernel while preserving correctness and public APIs."
tools: Bash, Read, Write, Edit, Glob, Grep
model: inherit
---

## 本项目实现硬约束

性能参考使用同语义CANN原生库或来源明确的Ascend C。优化候选的全部设备计算、复制、填充、归约、扫描、布局转换及回退必须由Triton kernel完成。

- `tl.load`／`tl.store`经Triton编译器生成DMA允许；宿主直接调用`aclrtMemcpyAsync`、CANN／aclnn目标算子、外部Ascend C设备核、PyTorch目标计算或CPU计算替代候选工作禁止。
- Python／C++只可承担分配、元数据、视图、编译缓存、参数、随机生成器状态和Triton启动。C++宿主桥须审查全部函数，并绑定候选、C++源码和二进制SHA；不得据此放行外部设备计算库。
- 此约束适用于分析建议、Coder修改、回退、评审、知识蒸馏和最佳候选保留。历史Ascend C／CANN／原生DMA成功经验只可作为参考侧的机制证据，不能作为候选代码模板。性能未达标时保留真实结果，继续优化Triton。
- 后端标签、导入Triton、KGS通过或性能高分均不足以证明合规。候选须经过绑定源码与依赖的逐路径审查和运行时门禁；正式交付还须通过原合同、补测、合格参考对照及实际编译路径核验。新代码和新分支必须重新审查。
- 保持原输入、容差、语义与计时合同；不从历史混合后端总数推导当前严格Triton达标数。可复用实现上下文取自本批`strict-implementation-context.json`，取用时重新核验源码与证据SHA；该文件中的参考实现单列，不参与候选择优。


你是一名资深 FlagGems 与 Triton 性能优化工程师，擅长从历史测量、kernel 结构和 workload 行为中提出可验证的单一优化假设。你每次调用只执行一轮优化，必须保持正确性与公共 API，并如实报告测试和 benchmark 结果。

When invoked:
1. 从动态输入读取目标算子与输出契约。
2. 阅读 `PERFORMANCE.md`、历史版本和当前 kernel，排除已尝试方向。
3. 选择并实施一个新的优化方向。
4. 反复修复正确性问题，正确性通过后只运行性能验证，不再叠加第二个方向。
5. 按动态输出契约返回本轮结构化结果并停止。

单轮优化检查清单：
- 历史记录和当前 kernel 已完整阅读
- 本轮方向未重复且只有一个
- tests/、benchmark/ 和公共 API 未修改
- 正确性测试全部通过后才接受性能数据
- speedup 来自实际 benchmark 输出
- 成功和失败都返回完整结构化结果

## 算子信息

- **算子名称**：以动态输入中的 `operator` 字段为准，下文用 `<operator>` 表示该值

## 工作流程

### 1. 查看性能历史

读取当前目录下的 `PERFORMANCE.md`（如果存在），了解已经尝试过的优化方向和对应的 speedup。

如果存在 `versions/` 目录，可以查看历史版本的 kernel 代码（`versions/vN/kernel.py`）了解之前的实现。

**重要**：不要重复已经尝试过的优化方向。

### 2. 分析并选择一个优化方向

根据性能历史和当前 kernel 代码，选择**一个**新的优化方向实施。

### 3. 修改 kernel 代码

编辑 kernel 文件实施优化。

### 4. 正确性验证

运行测试：

```bash
python -m pytest tests/ -m <operator> -vs 2>&1
```

**必须全部 PASS**。如果不通过，反复修复直到通过。

### 5. 性能验证

运行 benchmark：

```bash
python -m pytest benchmark/ -m <operator> -vs 2>&1
```

记录 speedup 数据。

### 6. 输出结果

完成后输出以下 JSON（用 ```json 代码块包裹）：

```json
{
  "operator": "<operator>",
  "status": "success 或 failed",
  "speedup": 0.0,
  "test_passed": true/false,
  "kernel_path": "相对 workspace 的 kernel 文件路径",
  "optimization_direction": "描述本轮优化方向",
  "error": null
}
```

## 规则

- **每轮只做一个优化方向**，不论性能是否提升都必须停止
- **正确性可以反复修复**直到通过，但通过后不要再为了性能修改代码
- **不要修改 tests/ 或 benchmark/**
- **公共 API 签名（函数名、参数列表）不可修改**
- **必须输出 JSON 结果**，即使失败也要输出
- **禁止 pip install**


# HG-KDA严格Triton夜间任务
{
  "operator": "expm1_",
  "current_seed": {
    "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/candidates/expm1_/a4ce60882923a4a0/main.py",
    "sha256": "a4ce60882923a4a045ba023dd9bbe9a860d263616f6bcba2af1ff0b176fec157"
  },
  "frozen_score": 0.6465574073403593,
  "frozen_source_sha256": "8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3",
  "per_case": [
    {
      "uuid": "expm1_-timing-0",
      "phase": "timing",
      "latency_ms": 0.011996408408532416,
      "reference_latency_ms": 0.010055278639305042
    },
    {
      "uuid": "expm1_-timing-1",
      "phase": "timing",
      "latency_ms": 0.012246074865445168,
      "reference_latency_ms": 0.010029605055142611
    },
    {
      "uuid": "expm1_-timing-2",
      "phase": "timing",
      "latency_ms": 0.013022450008248485,
      "reference_latency_ms": 0.010866799026209953
    },
    {
      "uuid": "expm1_-timing-3",
      "phase": "timing",
      "latency_ms": 0.05870354944901348,
      "reference_latency_ms": 0.022965065226617124
    },
    {
      "uuid": "expm1_-timing-4",
      "phase": "timing",
      "latency_ms": 0.01378702079205471,
      "reference_latency_ms": 0.011886630975862645
    },
    {
      "uuid": "expm1_-timing-5",
      "phase": "timing",
      "latency_ms": 0.014800413240342447,
      "reference_latency_ms": 0.011953646806062505
    },
    {
      "uuid": "expm1_-timing-6",
      "phase": "timing",
      "latency_ms": 0.013560554026716557,
      "reference_latency_ms": 0.011177804501617693
    },
    {
      "uuid": "expm1_-timing-7",
      "phase": "timing",
      "latency_ms": 0.09051665926093033,
      "reference_latency_ms": 0.025055513455611213
    },
    {
      "uuid": "expm1_-timing-8",
      "phase": "timing",
      "latency_ms": 0.012913621235209394,
      "reference_latency_ms": 0.010302411379293558
    },
    {
      "uuid": "expm1_-timing-9",
      "phase": "timing",
      "latency_ms": 0.013240622109201393,
      "reference_latency_ms": 0.01047869909903241
    },
    {
      "uuid": "expm1_-timing-10",
      "phase": "timing",
      "latency_ms": 0.012914128102713333,
      "reference_latency_ms": 0.010646626688179601
    },
    {
      "uuid": "expm1_-timing-11",
      "phase": "timing",
      "latency_ms": 0.0894069890800437,
      "reference_latency_ms": 0.02574595283894901
    }
  ],
  "paired_hardware_evidence": [
    {
      "dtype": "float16",
      "summary": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-float16/summary.json",
      "sha256": "cc5ff283d74f46dacd5a8c1e75defa54ba5f5f7de678e1c2ac68754a1fac7522",
      "spec": {
        "operator": "expm1_",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/starting-points/expm1_.py",
        "source_sha256": "8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3",
        "leaf": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/catalog/ops/expm1_",
        "case": {
          "name": "expm1_-timing-3",
          "inputs": {
            "spec": {
              "dtype": "float16",
              "shape": [
                4096,
                4096
              ]
            }
          },
          "seed": 1732,
          "tolerance": {
            "atol": 0.015,
            "rtol": 0.015,
            "required_matched_ratio": 1.0,
            "equal_nan": true
          }
        },
        "reference_kind": "original_PyTorch_only"
      },
      "differences": [
        {
          "table": "Memory.csv/vector0",
          "metric": "GM_to_UB_datas(KB)",
          "aggregation": "sum",
          "reference": 32768.0,
          "candidate": 32768.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "UB_to_GM_datas(KB)",
          "aggregation": "sum",
          "reference": 32768.0,
          "candidate": 32768.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte2_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 6184.0,
          "candidate_over_reference": 4.294444444444444,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte3_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 2088.0,
          "candidate_over_reference": 1.45,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 26.0437878,
          "candidate": 59.7665455,
          "candidate_over_reference": 2.2948484283073447,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte2_time(us)",
          "aggregation": "mean",
          "reference": 18.782242275,
          "candidate": 11.5750455,
          "candidate_over_reference": 0.6162760191527771,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte3_time(us)",
          "aggregation": "mean",
          "reference": 9.154363675,
          "candidate": 9.90481825,
          "candidate_over_reference": 1.0819777978724445,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_time(us)",
          "aggregation": "mean",
          "reference": 1.9265605749999999,
          "candidate": 3.96322725,
          "candidate_over_reference": 2.057151641857926,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_wait_time(us)",
          "aggregation": "mean",
          "reference": 0.0,
          "candidate": 55.4301358,
          "candidate_over_reference": null,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 26.0437878,
          "candidate": 59.7665455,
          "candidate_over_reference": 2.2948484283073447,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_vec_time(us)",
          "aggregation": "mean",
          "reference": 20.055273099999997,
          "candidate": 41.3362428,
          "candidate_over_reference": 2.0611159266636965,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        }
      ],
      "derived_differences": {
        "Memory.csv/vector0/bytes_per_mte2": {
          "reference": 23301.68888888889,
          "candidate": 5426.007761966365,
          "candidate_over_reference": 0.2328589909443726
        }
      },
      "actual_ir": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-float16/candidate/ir/manifest.json"
    },
    {
      "dtype": "bfloat16",
      "summary": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-bfloat16/summary.json",
      "sha256": "9ab36403a99e76ede4d2d64e9da1e45365931f940183712b5aa2c65e74241082",
      "spec": {
        "operator": "expm1_",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/starting-points/expm1_.py",
        "source_sha256": "8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3",
        "leaf": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/catalog/ops/expm1_",
        "case": {
          "name": "expm1_-timing-7",
          "inputs": {
            "spec": {
              "dtype": "bfloat16",
              "shape": [
                4096,
                4096
              ]
            }
          },
          "seed": 1736,
          "tolerance": {
            "atol": 0.015,
            "rtol": 0.015,
            "required_matched_ratio": 1.0,
            "equal_nan": true
          }
        },
        "reference_kind": "original_PyTorch_only"
      },
      "differences": [
        {
          "table": "Memory.csv/vector0",
          "metric": "GM_to_UB_datas(KB)",
          "aggregation": "sum",
          "reference": 32768.0,
          "candidate": 32768.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "UB_to_GM_datas(KB)",
          "aggregation": "sum",
          "reference": 32768.0,
          "candidate": 32768.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte2_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 6184.0,
          "candidate_over_reference": 4.294444444444444,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte3_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 2088.0,
          "candidate_over_reference": 1.45,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 26.409787775,
          "candidate": 98.25633305,
          "candidate_over_reference": 3.720451443498962,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte2_time(us)",
          "aggregation": "mean",
          "reference": 17.660878775,
          "candidate": 19.2931664,
          "candidate_over_reference": 1.0924239187525933,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte3_time(us)",
          "aggregation": "mean",
          "reference": 9.075500025,
          "candidate": 9.909878775,
          "candidate_over_reference": 1.091937496303406,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_time(us)",
          "aggregation": "mean",
          "reference": 1.7750606,
          "candidate": 4.4780302,
          "candidate_over_reference": 2.5227477867516184,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_wait_time(us)",
          "aggregation": "mean",
          "reference": 0.0,
          "candidate": 93.170939425,
          "candidate_over_reference": null,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 26.409787775,
          "candidate": 98.25633305,
          "candidate_over_reference": 3.720451443498962,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_vec_time(us)",
          "aggregation": "mean",
          "reference": 22.150303875000002,
          "candidate": 65.6621808,
          "candidate_over_reference": 2.964391873382369,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        }
      ],
      "derived_differences": {
        "Memory.csv/vector0/bytes_per_mte2": {
          "reference": 23301.68888888889,
          "candidate": 5426.007761966365,
          "candidate_over_reference": 0.2328589909443726
        }
      },
      "actual_ir": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-bfloat16/candidate/ir/manifest.json"
    },
    {
      "dtype": "float32",
      "summary": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-float32/summary.json",
      "sha256": "4d0d91eb06bb8885d0a28846a2614a7b188439e262ae70f298cf202190376e45",
      "spec": {
        "operator": "expm1_",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/starting-points/expm1_.py",
        "source_sha256": "8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3",
        "leaf": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/catalog/ops/expm1_",
        "case": {
          "name": "expm1_-timing-11",
          "inputs": {
            "spec": {
              "dtype": "float32",
              "shape": [
                4096,
                4096
              ]
            }
          },
          "seed": 1740,
          "tolerance": {
            "atol": 0.0001,
            "rtol": 0.0001,
            "required_matched_ratio": 1.0,
            "equal_nan": true
          }
        },
        "reference_kind": "original_PyTorch_only"
      },
      "differences": [
        {
          "table": "Memory.csv/vector0",
          "metric": "GM_to_UB_datas(KB)",
          "aggregation": "sum",
          "reference": 65536.0,
          "candidate": 65536.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "UB_to_GM_datas(KB)",
          "aggregation": "sum",
          "reference": 65536.0,
          "candidate": 65536.0,
          "candidate_over_reference": 1.0,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte2_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 12328.0,
          "candidate_over_reference": 8.561111111111112,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_mte3_instructions",
          "aggregation": "sum",
          "reference": 1440.0,
          "candidate": 4136.0,
          "candidate_over_reference": 2.8722222222222222,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "完整记录计数总和"
        },
        {
          "table": "Memory.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 44.511287975,
          "candidate": 93.2691513,
          "candidate_over_reference": 2.0954044590303726,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte2_time(us)",
          "aggregation": "mean",
          "reference": 42.325196925,
          "candidate": 22.8790455,
          "candidate_over_reference": 0.540553787393867,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_mte3_time(us)",
          "aggregation": "mean",
          "reference": 22.3758485,
          "candidate": 20.000545425,
          "candidate_over_reference": 0.8938452289306481,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_time(us)",
          "aggregation": "mean",
          "reference": 1.73565155,
          "candidate": 6.734469675,
          "candidate_over_reference": 3.880081618340962,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_scalar_wait_time(us)",
          "aggregation": "mean",
          "reference": 0.0,
          "candidate": 86.262575475,
          "candidate_over_reference": null,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_time(us)",
          "aggregation": "mean",
          "reference": 44.511287975,
          "candidate": 93.2691513,
          "candidate_over_reference": 2.0954044590303726,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        },
        {
          "table": "PipeUtilization.csv/vector0",
          "metric": "aiv_vec_time(us)",
          "aggregation": "mean",
          "reference": 12.86230275,
          "candidate": 57.4021814,
          "candidate_over_reference": 4.462823066421757,
          "reference_rows": 40,
          "candidate_rows": 40,
          "scope": "逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时"
        }
      ],
      "derived_differences": {
        "Memory.csv/vector0/bytes_per_mte2": {
          "reference": 46603.37777777778,
          "candidate": 5443.6132381570405,
          "candidate_over_reference": 0.11680726800778714
        }
      },
      "actual_ir": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/paired-best-float32/candidate/ir/manifest.json"
    }
  ],
  "independently_tested_proposals": [
    {
      "proposal": "codex-fma-series-v1",
      "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-fma-series-v1/main.py",
      "source_sha256": "fb69588d127ebdc1909224c74947e128c9b152e73430fdc83706b64fbedb03d0",
      "compilation": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/compile-fma-v1/result.json",
        "status": "compile_failed",
        "rows": [
          {
            "dtype": "fp16",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "fp16",
            "symbol": "_expm1_streaming2",
            "block": 8192,
            "status": "compile_failed",
            "ttir_fma_count": null
          },
          {
            "dtype": "bf16",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "bf16",
            "symbol": "_expm1_streaming1",
            "block": 8192,
            "status": "compile_failed",
            "ttir_fma_count": null
          },
          {
            "dtype": "fp32",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "fp32",
            "symbol": "_expm1_streaming2",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 2
          }
        ]
      }
    },
    {
      "proposal": "codex-typed-fma-v2",
      "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-typed-fma-v2/main.py",
      "source_sha256": "9db1896270d72bd68b0a541f1f07c39fea48c470d64bfb850f5f3774b37b7d75",
      "result": {
        "status": "evaluated",
        "source_sha256": "9db1896270d72bd68b0a541f1f07c39fea48c470d64bfb850f5f3774b37b7d75",
        "reference": "original_PyTorch_only",
        "CANN_accepted": false,
        "formal": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-typed-fma-v2/pytorch/strict-PyTorch-1.result.json",
        "formal_status": "PASSED",
        "PyTorch": 0.618491915932392,
        "supplement": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-typed-fma-v2/extra-pytorch.json",
        "exact": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-typed-fma-v2/exact-expm1-probe/result.json",
        "qualified_original": true,
        "full_fp16_domain": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/proposals/expm1_/codex-typed-fma-v2/full-fp16-domain/result.json",
        "improved": false,
        "reason": "低于冻结最佳，保留一次正式结果，不重测挑分"
      },
      "compilation": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/analysis/expm1_/compile-typed-fma-v2/result.json",
        "status": "compiled",
        "rows": [
          {
            "dtype": "fp16",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "fp16",
            "symbol": "_expm1_streaming2",
            "block": 8192,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 2
          },
          {
            "dtype": "bf16",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "bf16",
            "symbol": "_expm1_streaming1",
            "block": 8192,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "fp32",
            "symbol": "_expm1_flat",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 1
          },
          {
            "dtype": "fp32",
            "symbol": "_expm1_streaming2",
            "block": 4096,
            "status": "compiled_without_runtime",
            "ttir_fma_count": 2
          }
        ]
      }
    }
  ],
  "strict_contract": "所有设备计算和回退必须自包含Python+Triton；不使用CANN/Ascend C/PyTorch计算/原生DMA或外部.so。原地每次只执行一次；只可缓存编译及元数据，不缓存输入、输出、旧stream或结果。",
  "target": "先实质改善当前0.646557×的expm1完整调用，争取超过0.8×；CANN参考资格仍单列，不用不准确参考凑分。",
  "diagnosis": "新最佳三类型总GM读写与参考相同，Vector时间却分别约2.06/2.96/4.46倍；MTE时间未同比恶化。优先核对数学运算、类型扩展、UB存活、流水与同步，而不是直接归咎HBM带宽。",
  "compiler_observation": "显式tl.fma(x,0.5,1.0)的默认常量在TTIR将FP16扩展为FP32，使FP16流式需求1835264 bits超过1572864、BF16也溢出；同dtype常量提案及实际结果在independently_tested_proposals。不能把更少源代码运算等同于更少设备指令。",
  "search_directions": [
    "读取实际TTIR和adapter/后续IR，看Vector指令和缓冲是否真正减少，结合已成功或失败的同dtype常量提案。不要重新完整评测同一代码。",
    "BF16会升为FP32执行部分运算，和FP16分别选择组织方式；检查混合精度、块内复用、同步、每核循环及软件流水。",
    "只在新的机制证据支持时试编译器num_stages、multibuffer、enable_ubuf_saving等正常选项，实际UB编译和数值先过，不能关闭必要同步或换后端。",
    "保留小值相对精度、负零、NaN/Inf、stride、尾部、非对齐、当前stream及冷启动只执行一次；不能直接删掉稳定公式以追赶不准确的参考。"
  ],
  "previous_failures": "上批纯FP16数学0.542×，仅消除比较扩展0.486×，都不优于0.647×；不要重复旧分块和同源码挑分。各dtype两块/单块的旧数值只是历史事实，不能写进新稿注释充当本轮测量。",
  "workflow": "KG/HG-KDA与CC+DeepSeek生成。Codex提供新配对指标、IR机制判断及独立提案；不是纯原生KG独立效果。先给出可运行的代码改进并完整evaluate，单次debug<=120秒，沿用现有诊断预算。"
}
