---
name: kernel-profile-analyzer
description: "Use this agent immediately after a fully passing KernelGen eval round requires profile analysis; select representative workloads, collect backend-native evidence, persist a validated analysis, and return one actionable experiment to the Coder."
capabilities: [read, search]
mcp_tools: [get_profile_context, profile_workloads, record_profile_analysis]
subagents: []
model: inherit
approval: no_prompts
---

You are a senior GPU performance analyst. You analyze one already-evaluated kernel round in an isolated subagent context, collect factual profiler evidence through backend-neutral tools, persist one validated ProfileAnalysis, and return a compact result to the parent Coder. You do not edit kernels, run evaluations, or invent measurements.

When invoked:
1. Extract only `round_num` from the delegation request and call `mcp__kernelgen__get_profile_context` before making any claim; the context includes the frozen experiment plan, Python-computed comparison, and complete workload measurements.
2. Verify that the authoritative eval status is `PASSED`, the evaluation fingerprint is present, and the profile state is `pending` or `collecting`.
3. Select workloads only from `profile_workload_uuids`, using their entries in the complete `per_workload` set as evidence. In legacy mode this list contains every workload; in phased mode it contains only timing workloads. Do not use legacy primary/repr concepts or impose an arbitrary count limit.
4. Start with `metrics`, read the downloaded reports, and request `source` or `instruction` only when finer localization is needed and advertised by service capabilities.
5. Separate profiler facts from your inference, preserve workload scope, and derive one focused next experiment for the parent Coder.
6. Always call `mcp__kernelgen__record_profile_analysis`, including when profiling is unsupported, fails, or remains inconclusive.
7. Return only after the record tool reports `recorded: true`; otherwise correct the payload and retry.

## Workload Selection

Among `profile_workload_uuids`, begin with the lowest-speedup workload and the highest-latency workload when they differ. Add eligible boundary workloads where axes or speedup change sharply, representatives of distinct performance regimes, and a strong control workload when it helps distinguish hypotheses. You may profile more workloads after reading the first reports. Stop when the evidence explains the important regimes, additional workloads no longer change the dominant finding, or the profiler cannot provide stronger evidence.

Never generalize one workload's bottleneck to the full set. Use `dominant_bound: mixed` and workload-scoped findings when different regimes exhibit different limits.

## Profile Levels

- `metrics` is the default first request and should provide the broad counters and normalized readable report.
- `source` is appropriate when the next source change depends on locating a hotspot or source line.
- `instruction` is appropriate when the hypothesis depends on SASS, Ascend simulator instructions, spill/load-store width, tensor/cube instructions, pipeline mapping, or source-to-instruction evidence.

Do not assume CUDA artifacts on Ascend or Ascend artifacts on CUDA. Inspect `service.profile.capabilities`, each result's `capabilities`, artifact `kind`, and artifact `local_path`. CUDA may expose NCU reports and SASS; Ascend may expose msprof reports and simulator instruction mappings. Other backends may be explicitly unsupported.

## Evidence Rules

- Eval speedup, candidate latency, reference latency, and before/after deltas come only from `get_profile_context`; profiler timing is diagnostic and never replaces eval latency.
- Every finding requires at least one downloaded artifact path and a metric or fact that can be located in that artifact.
- Put direct measurements in `evidence`; put causal interpretation in `inference` and calibrate confidence as `high`, `medium`, or `low`.
- Use the exact evaluation fingerprint, solution SHA-256, backend, workload UUIDs, profile IDs, manifest paths, and artifact paths returned by the tools.
- Reports contain facts, not optimization advice. You are responsible for the inference and must not present an unsupported suggestion as measured truth.

## Terminal Outcomes

- Use `completed` only when at least one workload was profiled, at least one evidence-backed finding is supported, and one concrete next experiment is specified.
- Use `inconclusive` when artifacts were collected but do not support a stable optimization conclusion; state what was collected and why it is insufficient.
- Use `unsupported` when the service declares profiling unavailable; preserve the reported capabilities or error.
- Use `failed` for transport, collection, download, or parsing failure; preserve the error and any successfully downloaded partial evidence.

## Next Experiment

Recommend exactly one focused change category and action. State expected impact, risks and rollback, validation workloads, and success criteria that include full eval correctness and headline `geo_mean`. Explicitly say whether the profiler evidence supports or contradicts the frozen hypothesis. Prefer a causal experiment that can falsify the dominant hypothesis over broad parameter sweeping.

## Return to Coder

After `record_profile_analysis` returns `recorded: true`, return a compact message containing `recorded`, `round_num`, `status`, `dominant_bound`, the main workload-scoped finding, the next experiment, and the saved analysis path. Do not inline full reports or large metric tables because the validated analysis and artifacts remain in the workspace.


# HG-KDA本批分析约束
结合相同输入的实际执行路径和指标比较，先看绝对值/差值/比率与记录范围，再查源码、IR和热点区域。高活跃比例只是待验证方向。get_profile_context的hgkda_workloads提供准确Native输入，不猜测case规格。优先使用当前结构化证据与作用域兼容知识；历史失败约束按需读取，不将全部历史知识注入上下文。保留原角色工具和正式KGS评测。


# 新25项任务
本轮目标native_group_norm：先修正确性，再用实际输入和profiling改进性能。避免只为用例列表做特化；保留原地/out、NaN/Inf、stride、广播和随机机制。复杂算子检查片上缓冲、归约、同步和布局。每轮记录具体预测与验证。不得调用PyTorch计算目标替代候选；CANN参考只用于对照。先提交第一份可运行的完整KGS候选；首轮正式评测前最多两次诊断debug，每次不超过120秒。不要在尚无完整候选时反复测启动微基准。
当前证据：
{
  "schema": "hgkda-next-context-v1",
  "spec": {
    "operator": "native_group_norm",
    "leaf": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/catalog/ops/native_group_norm",
    "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
    "source_sha256": "12d25c47d259a2dad3ef596cd631d118fc649fd68e15333dd32b28529e672baa",
    "case": {
      "name": "native_group_norm-timing-0",
      "inputs": {
        "spec": {
          "dtype": "float16",
          "shape": [
            4,
            32,
            32,
            32
          ],
          "groups": 8,
          "affine": true
        }
      },
      "seed": 1729,
      "tolerance": {
        "atol": 0.015,
        "rtol": 0.015,
        "required_matched_ratio": 1.0,
        "equal_nan": true
      }
    },
    "scope": "新批起点代表输入；不冒充新候选结果"
  },
  "status": "partial",
  "paths": {
    "candidate": {
      "status": "completed",
      "kernels": [
        {
          "name": "_gn_kernel",
          "basic": {
            "Op Name": "_gn_kernel",
            "Op Type": "vector",
            "Task Duration(us)": "6.560131",
            "Block Dim": "32",
            "Mix Block Dim": "NA",
            "Device Id": "7",
            "Pid": "1245932",
            "Current Freq": "1650",
            "Rated Freq": "1650",
            "": ""
          }
        }
      ],
      "error": null,
      "metrics": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/agent-context-candidate-path-metrics.json",
        "sha256": "0e154bb7136ffaf6eb9f122372be5e2c718dd41a254c9fdf48886371c9bc3151",
        "kernels": [
          {
            "kernel": "_gn_kernel",
            "selected_metrics": [
              {
                "table": "ArithmeticUtilization.csv/vector0",
                "metric": "aic_cube_fops",
                "value": null,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "ArithmeticUtilization.csv/vector0",
                "metric": "aic_cube_fp16_ratio",
                "value": null,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "ArithmeticUtilization.csv/vector0",
                "metric": "aic_cube_fp_instr_number",
                "value": null,
                "rows": 32,
                "aggregation": "sum"
              },
              {
                "table": "ArithmeticUtilization.csv/vector0",
                "metric": "aic_cube_int8_ratio",
                "value": null,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "Memory.csv/vector0",
                "metric": "GM_to_L1_datas(KB)",
                "value": null,
                "rows": 32,
                "aggregation": "sum"
              },
              {
                "table": "Memory.csv/vector0",
                "metric": "GM_to_UB_datas(KB)",
                "value": 512.0,
                "rows": 32,
                "aggregation": "sum"
              },
              {
                "table": "Memory.csv/vector0",
                "metric": "L0C_to_GM_datas(KB)",
                "value": null,
                "rows": 32,
                "aggregation": "sum"
              },
              {
                "table": "Memory.csv/vector0",
                "metric": "L0C_to_L1_datas(KB)",
                "value": null,
                "rows": 32,
                "aggregation": "sum"
              },
              {
                "table": "PipeUtilization.csv/vector0",
                "metric": "aic_scalar_wait_ib_time(us)",
                "value": null,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "PipeUtilization.csv/vector0",
                "metric": "aic_scalar_wait_time(us)",
                "value": null,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "PipeUtilization.csv/vector0",
                "metric": "aiv_scalar_wait_ib_time(us)",
                "value": 0.0159095,
                "rows": 32,
                "aggregation": "mean"
              },
              {
                "table": "PipeUtilization.csv/vector0",
                "metric": "aiv_scalar_wait_time(us)",
                "value": 0.97630675,
                "rows": 32,
                "aggregation": "mean"
              }
            ],
            "all_metric_count": 156,
            "granularity": {
              "Memory.csv/vector0/GM_to_UB_datas(KB)/aiv_mte2_instructions": {
                "bytes_per_instruction": 712.3478260869565,
                "traffic_kb": 512.0,
                "instructions": 736.0,
                "interpretation": "厂商CSV接口流量与完整计数之比，沿用旧工具KB×1024约定；不是物理HBM事务大小"
              },
              "Memory.csv/vector0/UB_to_GM_datas(KB)/aiv_mte3_instructions": {
                "bytes_per_instruction": 1206.857142857143,
                "traffic_kb": 264.0,
                "instructions": 224.0,
                "interpretation": "厂商CSV接口流量与完整计数之比，沿用旧工具KB×1024约定；不是物理HBM事务大小"
              },
              "Memory.csv/vector0/GM_to_L1_datas(KB)/aic_mte2_instructions": {
                "bytes_per_instruction": null,
                "traffic_kb": null,
                "instructions": null,
                "interpretation": "厂商CSV接口流量与完整计数之比，沿用旧工具KB×1024约定；不是物理HBM事务大小"
              }
            }
          }
        ],
        "aggregate_work": [
          {
            "table": "ArithmeticUtilization.csv/vector0",
            "metric": "aic_cube_fp_instr_number",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "ArithmeticUtilization.csv/vector0",
            "metric": "aic_cube_int_instr_number",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "ArithmeticUtilization.csv/vector0",
            "metric": "aic_cube_total_instr_number",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "GM_to_L1_datas(KB)",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "GM_to_UB_datas(KB)",
            "value": 512.0,
            "observed_sum": 512.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "L0C_to_GM_datas(KB)",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "L0C_to_L1_datas(KB)",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "L1_to_GM_datas(KB)(estimate)",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "UB_to_GM_datas(KB)",
            "value": 264.0,
            "observed_sum": 264.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "aic_mte1_instructions",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "aic_mte2_instructions",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "aic_mte3_instructions",
            "value": null,
            "observed_sum": null,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "aiv_mte2_instructions",
            "value": 736.0,
            "observed_sum": 736.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "aiv_mte3_instructions",
            "value": 224.0,
            "observed_sum": 224.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "read_main_memory_datas(KB)",
            "value": 528.0,
            "observed_sum": 528.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          },
          {
            "table": "Memory.csv/vector0",
            "metric": "write_main_memory_datas(KB)",
            "value": 264.0,
            "observed_sum": 264.0,
            "kernels": 1,
            "total_kernels": 1,
            "complete": true
          }
        ],
        "scope": "逐kernel时间保持各自统计范围；完整流量和指令计数可加。不将跨核时间均值相加当作端到端关键路径。"
      },
      "semantics": {
        "status": "passed",
        "reference": null
      }
    },
    "pytorch": {
      "status": "failed",
      "kernels": [],
      "error": "AssertionError: 基本信息不唯一",
      "semantics": {
        "status": "passed",
        "reference": null
      }
    },
    "cann": {
      "status": "failed",
      "kernels": [],
      "error": "AssertionError: 基本信息不唯一",
      "semantics": {
        "status": "passed",
        "reference": {
          "operator": "native_group_norm",
          "reference_kind": "CANN_native_aclnn",
          "query": "aclnnGroupNormGetWorkspaceSize",
          "execute": "aclnnGroupNorm",
          "dtypes": [
            "float16",
            "bfloat16",
            "float32"
          ],
          "effects": "输入不变，返回out/mean/rstd",
          "source_implementation": "库内实现未核验，不标作已获得Ascend C源码",
          "header": "/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_group_norm.h",
          "header_sha256": "5d56c5ab9723c0c89218755132eb7a887f99715c450c672a2049a67e3d97a1b4",
          "library": "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libopapi.so",
          "library_sha256": "f337f61d9bcdf87421c8be77f906d492ad0c782114bd09a92c0c747fd9bcada2",
          "binding_source_sha256": "4d9c02aceec36cb1de57046849aa0d6501d6beee3eafe53793877aeae16a6af9",
          "scope": "直接aclnn两段接口；PyTorch仅分配张量/获取stream，生命周期同步导致不可直接作为完整调用速度成绩"
        }
      }
    }
  },
  "comparisons": [],
  "regions": [
    {
      "schema": "hgkda-regions-v1",
      "status": "completed",
      "regions": [
        {
          "id": "_gn_kernel|core0.veccore0|('source_neighborhood', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py', 73, 80)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "source_neighborhood",
          "file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
          "start": 73,
          "end": 80,
          "instructions": 347,
          "pipelines": [
            "FLOWCTRL",
            "SCALAR",
            "VECTOR"
          ],
          "instruction_time_sum_us": 4.2199999999999935,
          "largest_instruction_us": 0.77,
          "wait_instruction_time_sum_us": 3.21,
          "pcs": [
            "0x10d111fc",
            "0x10d1123c",
            "0x10d1121c",
            "0x10d11274",
            "0x10d1127c",
            "0x10d1124c",
            "0x10d1120c",
            "0x10d1122c",
            "0x10d116a0",
            "0x10d11424",
            "0x10d115ec",
            "0x10d11370",
            "0x10d115dc",
            "0x10d11360",
            "0x10d1135c",
            "0x10d115d8"
          ],
          "source_lines": [
            74,
            75,
            78,
            79,
            80
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 1.0839902580482885,
          "trace_time_unit": "us",
          "pc_count": 347
        },
        {
          "id": "_gn_kernel|core0.veccore0|('source_neighborhood', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py', 97, 104)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "source_neighborhood",
          "file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
          "start": 97,
          "end": 104,
          "instructions": 47,
          "pipelines": [
            "MTE2",
            "MTE3",
            "SCALAR",
            "VECTOR"
          ],
          "instruction_time_sum_us": 7.609999999999999,
          "largest_instruction_us": 0.8,
          "wait_instruction_time_sum_us": 4.07,
          "pcs": [
            "0x10d119c0",
            "0x10d11998",
            "0x10d11978",
            "0x10d1194c",
            "0x10d11904",
            "0x10d11908",
            "0x10d119c8",
            "0x10d119d0",
            "0x10d119c4",
            "0x10d119bc",
            "0x10d1190c",
            "0x10d11900",
            "0x10d118fc",
            "0x10d11910",
            "0x10d1195c",
            "0x10d11928"
          ],
          "source_lines": [
            102,
            103
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 0.9980334327556193,
          "trace_time_unit": "us",
          "pc_count": 47
        },
        {
          "id": "_gn_kernel|core0.veccore0|('source_neighborhood', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py', 65, 72)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "source_neighborhood",
          "file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
          "start": 65,
          "end": 72,
          "instructions": 71,
          "pipelines": [
            "FLOWCTRL",
            "MTE2",
            "SCALAR",
            "VECTOR"
          ],
          "instruction_time_sum_us": 4.769999999999998,
          "largest_instruction_us": 0.76,
          "wait_instruction_time_sum_us": 1.79,
          "pcs": [
            "0x10d111cc",
            "0x10d111c0",
            "0x10d111c4",
            "0x10d111c8",
            "0x10d1126c",
            "0x10d111bc",
            "0x10d111b8",
            "0x10d11064",
            "0x10d1106c",
            "0x10d11078",
            "0x10d11084",
            "0x10d11140",
            "0x10d11144",
            "0x10d111dc",
            "0x10d111f8",
            "0x10d11114"
          ],
          "source_lines": [
            65,
            70
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 0.8332414850592613,
          "trace_time_unit": "us",
          "pc_count": 71
        },
        {
          "id": "_gn_kernel|core0.veccore0|('source_neighborhood', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py', 89, 96)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "source_neighborhood",
          "file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
          "start": 89,
          "end": 96,
          "instructions": 45,
          "pipelines": [
            "FLOWCTRL",
            "MTE2",
            "MTE3",
            "SCALAR",
            "VECTOR"
          ],
          "instruction_time_sum_us": 1.5100000000000002,
          "largest_instruction_us": 0.44,
          "wait_instruction_time_sum_us": 0.67,
          "pcs": [
            "0x10d119d8",
            "0x10d118e8",
            "0x10d118f4",
            "0x10d119e0",
            "0x10d118a8",
            "0x10d118b4",
            "0x10d118f8",
            "0x10d11880",
            "0x10d118f0",
            "0x10d11968",
            "0x10d11930",
            "0x10d11944",
            "0x10d1188c",
            "0x10d11924",
            "0x10d11960",
            "0x10d11884"
          ],
          "source_lines": [
            89,
            90,
            94,
            95
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 0.7805017794016749,
          "trace_time_unit": "us",
          "pc_count": 45
        },
        {
          "id": "_gn_kernel|core0.veccore0|('pc_neighborhood', 'unmapped', 282136640, 282136703)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "pc_neighborhood",
          "file": "unmapped",
          "start": 282136640,
          "end": 282136703,
          "instructions": 8,
          "pipelines": [
            "SCALAR"
          ],
          "instruction_time_sum_us": 0.78,
          "largest_instruction_us": 0.22,
          "wait_instruction_time_sum_us": 0,
          "pcs": [
            "0x10d11044",
            "0x10d11048",
            "0x10d11050",
            "0x10d11058",
            "0x10d1105c",
            "0x10d11054",
            "0x10d1104c",
            "0x10d11040"
          ],
          "source_lines": [
            0
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 0.39699944853782654,
          "trace_time_unit": "us",
          "pc_count": 8
        },
        {
          "id": "_gn_kernel|core0.veccore0|('source_neighborhood', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py', 81, 88)",
          "kernel": "_gn_kernel",
          "core": "core0.veccore0",
          "kind": "source_neighborhood",
          "file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
          "start": 81,
          "end": 88,
          "instructions": 27,
          "pipelines": [
            "MTE2",
            "MTE3",
            "SCALAR"
          ],
          "instruction_time_sum_us": 0.32000000000000006,
          "largest_instruction_us": 0.14,
          "wait_instruction_time_sum_us": 0.0,
          "pcs": [
            "0x10d11854",
            "0x10d11870",
            "0x10d1182c",
            "0x10d11830",
            "0x10d11860",
            "0x10d1185c",
            "0x10d11858",
            "0x10d1183c",
            "0x10d11838",
            "0x10d1186c",
            "0x10d117c8",
            "0x10d1184c",
            "0x10d11844",
            "0x10d11840",
            "0x10d11828",
            "0x10d11824"
          ],
          "source_lines": [
            82,
            83,
            84
          ],
          "contains_function_entry": false,
          "interpretation": "源码邻域/PC邻域；累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果",
          "trace_interval_union": 0.16650187433697283,
          "trace_time_unit": "us",
          "pc_count": 27
        }
      ],
      "total_regions": 11,
      "timeline": {
        "status": "completed",
        "core": "core0.veccore0",
        "kernel": "_gn_kernel",
        "time_unit": "us",
        "capture_start": 112.41899871826172,
        "capture_end": 115.65599877934437,
        "capture_duration": 3.2370000610826537,
        "capture_scope": "本核硬件流水首个至末个可见事件；无硬件泳道时使用全部事件并单列；不包含未知启动前后空闲",
        "display_time_unit": "ns",
        "pipelines": [
          {
            "pipeline": "ALL",
            "event_count": 1,
            "active_union": 0.0010000000474974513,
            "active_ratio": 0.0003089280286151707,
            "wait_union": 0.0010000000474974513,
            "wait_ratio": 0.0003089280286151707,
            "idle_union": 3.2360000610351562,
            "idle_ratio": 0.9996910719713848,
            "productive_union": 0,
            "productive_ratio": 0.0,
            "wait_only_union": 0.0010000000474974513,
            "wait_only_ratio": 0.0003089280286151707,
            "pipeline_kind": "auxiliary_trace_lane",
            "largest_idle": 3.2350006103515625,
            "idle_intervals": [
              [
                112.41899871826172,
                115.65399932861328
              ],
              [
                115.65499932866078,
                115.65599877934437
              ]
            ],
            "idle_intervals_total": 2,
            "activity_includes_wait": true
          },
          {
            "pipeline": "CACHEMISS",
            "event_count": 28,
            "active_union": 0.020993957994505763,
            "active_ratio": 0.0064856217480218645,
            "wait_union": 0,
            "wait_ratio": 0.0,
            "idle_union": 3.216006103088148,
            "idle_ratio": 0.9935143782519781,
            "productive_union": 0.020993957994505763,
            "productive_ratio": 0.0064856217480218645,
            "wait_only_union": 0,
            "wait_only_ratio": 0.0,
            "pipeline_kind": "auxiliary_trace_lane",
            "largest_idle": 0.996002197265625,
            "idle_intervals": [
              [
                112.41899871826172,
                112.6969985961914
              ],
              [
                112.69999749760609,
                113.59500122070312
              ],
              [
                113.59800012211781,
                113.61799621582031
              ],
              [
                113.61899621586781,
                113.61900329589844
              ],
              [
                113.62100274662953,
                113.94999694824219
              ],
              [
                113.95199639897328,
                113.9520034790039
              ],
              [
                113.9530034790514,
                114.0
              ],
              [
                114.00299890141468,
                114.43599700927734
              ]
            ],
            "idle_intervals_total": 11,
            "activity_includes_wait": false
          },
          {
            "pipeline": "FLOWCTRL",
            "event_count": 1,
            "active_union": 5.493639037013054e-07,
            "active_ratio": 1.6971389970180106e-07,
            "wait_union": 0,
            "wait_ratio": 0.0,
            "idle_union": 3.23699951171875,
            "idle_ratio": 0.9999998302861003,
            "productive_union": 5.493639037013054e-07,
            "productive_ratio": 1.6971389970180106e-07,
            "wait_only_union": 0,
            "wait_only_ratio": 0.0,
            "pipeline_kind": "auxiliary_trace_lane",
            "largest_idle": 3.23699951171875,
            "idle_intervals": [
              [
                112.41899871826172,
                115.65599822998047
              ]
            ],
            "idle_intervals_total": 1,
            "activity_includes_wait": false
          },
          {
            "pipeline": "MTE2",
            "event_count": 37,
            "active_union": 1.5051527894102037,
            "active_ratio": 0.46498386191157104,
            "wait_union": 0.3981818091124296,
            "wait_ratio": 0.12300951547688661,
            "idle_union": 1.73184727167245,
            "idle_ratio": 0.535016138088429,
            "productive_union": 1.2102184818359092,
            "productive_ratio": 0.37387039202932143,
            "wait_only_union": 0.2949343075742945,
            "wait_only_ratio": 0.09111346988224961,
            "pipeline_kind": "hardware_execution",
            "largest_idle": 0.9487561956048012,
            "idle_intervals": [
              [
                112.41899871826172,
                112.7036361694336
              ],
              [
                112.71696950308979,
                112.71800231933594
              ],
              [
                113.48424246162176,
                114.43299865722656
              ],
              [
                114.43499865732156,
                114.447998046875
              ],
              [
                114.44999804697,
                114.65599822998047
              ],
              [
                115.37757754325867,
                115.65599877934437
              ]
            ],
            "idle_intervals_total": 6,
            "activity_includes_wait": true
          },
          {
            "pipeline": "MTE3",
            "event_count": 20,
            "active_union": 1.0994766515213996,
            "active_ratio": 0.33965913832997163,
            "wait_union": 0.4702120269648731,
            "wait_ratio": 0.14526166762184273,
            "idle_union": 2.137523409561254,
            "idle_ratio": 0.6603408616700284,
            "productive_union": 0.8758471774635836,
            "productive_ratio": 0.27057372905042393,
            "wait_only_union": 0.22362947405781597,
            "wait_only_ratio": 0.06908540927954768,
            "pipeline_kind": "hardware_execution",
            "largest_idle": 2.0190048217773438,
            "idle_intervals": [
              [
                112.41899871826172,
                114.43800354003906
              ],
              [
                114.58848142682109,
                114.70303344726562
              ],
              [
                115.50303220807109,
                115.50399780273438
              ],
              [
                115.65299780666828,
                115.65599877934437
              ]
            ],
            "idle_intervals_total": 4,
            "activity_includes_wait": true
          },
          {
            "pipeline": "SCALAR",
            "event_count": 49,
            "active_union": 0.8789328017737716,
            "active_ratio": 0.27152696484034106,
            "wait_union": 0.6127272701123729,
            "wait_ratio": 0.1892886186438436,
            "idle_union": 2.358067259308882,
            "idle_ratio": 0.7284730351596589,
            "productive_union": 0.2662055316613987,
            "productive_ratio": 0.08223834619649746,
            "wait_only_union": 0.6127272701123729,
            "wait_only_ratio": 0.1892886186438436,
            "pipeline_kind": "hardware_execution",
            "largest_idle": 0.86199951171875,
            "idle_intervals": [
              [
                112.41899871826172,
                112.4260025024414
              ],
              [
                112.64700250327587,
                112.65699768066406
              ],
              [
                112.66099713137373,
                112.7020034790039
              ],
              [
                112.7030034790514,
                112.70999908447266
              ],
              [
                112.71099908452015,
                112.93399810791016
              ],
              [
                112.93499810795765,
                112.94200134277344
              ],
              [
                112.94300134282093,
                112.95899963378906
              ],
              [
                112.95999963383656,
                112.96700286865234
              ]
            ],
            "idle_intervals_total": 38,
            "activity_includes_wait": true
          },
          {
            "pipeline": "SCALARLDST",
            "event_count": 859,
            "active_union": 1.18688445573207,
            "active_ratio": 0.36666185768779447,
            "wait_union": 0,
            "wait_ratio": 0.0,
            "idle_union": 2.050115605350584,
            "idle_ratio": 0.6333381423122055,
            "productive_union": 1.18688445573207,
            "productive_ratio": 0.36666185768779447,
            "wait_only_union": 0,
            "wait_only_ratio": 0.0,
            "pipeline_kind": "hardware_execution",
            "largest_idle": 0.8479990233900025,
            "idle_intervals": [
              [
                112.42499652109109,
                112.42500305175781
              ],
              [
                112.42600305180531,
                112.427001953125
              ],
              [
                112.6470019519329,
                112.64800262451172
              ],
              [
                112.8300007879734,
                112.91500091552734
              ],
              [
                112.91700036625843,
                112.91899871826172
              ],
              [
                112.92099816899281,
                112.9219970703125
              ],
              [
                112.92299707036,
                112.92400360107422
              ],
              [
                112.92500360112172,
                112.9260025024414
              ]
            ],
            "idle_intervals_total": 97,
            "activity_includes_wait": false
          },
          {
            "pipeline": "VECTOR",
            "event_count": 138,
            "active_union": 2.206257200217806,
            "active_ratio": 0.6815746551082537,
            "wait_union": 1.1084911478683352,
            "wait_ratio": 0.3424439687831167,
            "idle_union": 1.030742860864848,
            "idle_ratio": 0.31842534489174634,
            "productive_union": 1.1189617057098076,
            "productive_ratio": 0.34567861742194633,
            "wait_only_union": 1.0872954945079982,
            "wait_only_ratio": 0.3358960376863073,
            "pipeline_kind": "hardware_execution",
            "largest_idle": 0.21699737571179867,
            "idle_intervals": [
              [
                112.41899871826172,
                112.42400360107422
              ],
              [
                112.4340036008507,
                112.6510009765625
              ],
              [
                112.66100097633898,
                112.66600036621094
              ],
              [
                112.67600036598742,
                112.68499755859375
              ],
              [
                112.7039975579828,
                112.7040023803711
              ],
              [
                113.0260015259264,
                113.0270004272461
              ],
              [
                113.083003051579,
                113.0830307006836
              ],
              [
                113.38199670414906,
                113.38200378417969
              ]
            ],
            "idle_intervals_total": 46,
            "activity_includes_wait": true
          }
        ],
        "outside_events": 4,
        "interpretation": "productive排除WAIT/BAR；wait_only排除与有效指令重叠的等待；productive+wait_only+idle=窗口。idle不是因果停顿，不重建关键路径；辅助trace泳道另标。Chrome ts/dur通常为微秒，displayTimeUnit仅显示单位，输入单位由调用者确认。"
      },
      "grouping": {
        "source_line_window": 8,
        "pc_bytes": 64,
        "top": 6
      },
      "limitations": [
        "源码邻域不是自动识别的循环/基本块边界；结合实际IR由Agent解释",
        "不同区域的区间也可能重叠，不累加为总耗时",
        "函数入口调试行号可能覆盖多段指令，已显式标记"
      ]
    }
  ],
  "execution_guidance": "优先尽早提交第一份完整KGS候选；首轮正式评测前最多两个诊断debug，每次不超过120秒。编译/精度问题先修复；不在未经正式评测的候选上连续做多轮启动微基准。使用真实指标/IR回答明确问题后修改并验证。",
  "limits": [
    "PyTorch仍为正式正确性/速度参考；CANN直接入口含生命周期同步，不把此处host开销作为速度成绩",
    "多kernel不按名称相似性盲目配对；只有两侧各一个kernel时输出逐指标比率",
    "数值Roofline不是每个复杂算子必须计算的前置条件，优先使用真实对照"
  ],
  "actual_ir": {
    "spec": {
      "operator": "native_group_norm",
      "leaf": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/catalog/ops/native_group_norm",
      "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
      "source_sha256": "12d25c47d259a2dad3ef596cd631d118fc649fd68e15333dd32b28529e672baa",
      "case": {
        "name": "native_group_norm-timing-0",
        "inputs": {
          "spec": {
            "dtype": "float16",
            "shape": [
              4,
              32,
              32,
              32
            ],
            "groups": 8,
            "affine": true
          }
        },
        "seed": 1729,
        "tolerance": {
          "atol": 0.015,
          "rtol": 0.015,
          "required_matched_ratio": 1.0,
          "equal_nan": true
        }
      },
      "scope": "新批起点代表输入；不冒充新候选结果"
    },
    "entries": [
      {
        "symbol": "_gn_kernel",
        "device": 0,
        "metadata": "KernelMetadata(add_auto_scheduling=False, allow_fp8e4nv=False, allowed_dot_input_precisions=['ieee', 'hf32'], arch='Ascend910B4-1', auto_blockify_size=1, auto_tile_and_bind_subblock=True, auto_vectorize_v2_max_fused_ops_num=None, bisheng_options='-cce-link-aicore-ll-module /usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', bitcodes=[], bs_task_type=10, cluster_dims=(1, 1, 1), compile_mode='simd', compile_on_910_95=False, custom_pipeline=None, debug=False, default_dot_input_precision='ieee', deprecated_fp8_dtypes=[], disable_auto_inject_block_sync=None, disable_fma=False, disable_size_align_for_cast=None, disable_tightly_coupled_buffer_reuse=False, enable_auto_bind_sub_block=None, enable_auto_blockify=None, enable_auto_vectorize_v2=None, enable_bishengir_simt_optimization=0, enable_cce_vf_auto_sync=None, enable_cce_vf_remove_membar=None, enable_drop_unit_dims=None, enable_dynamic_cv_pipeline=False, enable_flatten=None, enable_fp_fusion=True, enable_hivm_auto_cv_balance=None, enable_mask_fallback_conversion=False, enable_mixed_cv=None, enable_nd2nz_on_vector=False, enable_persistent=False, enable_preload=None, enable_select_analysis=True, enable_simt_reorder_instruction=False, enable_sync_block_lock=False, enable_ubuf_saving=None, enable_vf_fusion=False, enable_warp_specialization=False, extern_libs=None, force_simt_only=False, force_simt_template=False, hash='7f745f419ce2d615c6222701baad7bb0d3c5cb5985d4b9a6e6a7665a136fc0e5', hfusion_enable_multiple_consumer_fusion=False, inject_barrier_all=None, inject_block_all=None, instrumentation_mode='', kernel_name='_gn_kernel', limit_auto_multi_buffer_of_local_buffer=None, limit_auto_multi_buffer_only_for_local_buffer=None, llvm_version=15, max_num_imprecise_acc_default=0, mix_mode='aiv', multibuffer=True, name='_gn_kernel', num_buffers_warp_spec=0, num_consumer_groups=0, num_ctas=1, num_stages=2, num_warps=32, optimize_dynamic_offset=False, optimize_epilogue=False, parallel_mode='simd', prevec_max_fused_ops_num=None, reg_dec_producer=0, reg_inc_consumer=0, required_ub_bits=0, sanitize_overflow=True, set_workspace_multibuffer=None, shared=1, shared_mem_dynamic_size=221184, simt_stack_limit=None, stream=None, supported_fp8_dtypes=['fp8e5', 'fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5b16'], sync_solver=None, target=GPUTarget(backend='npu', arch='Ascend910B4-1', warp_size=0), tensor_kinds=[0, 1, 1, 1, 0, 0], tile_mix_cube_loop=None, tile_mix_vector_loop=None, triton_version='3.5.1', unit_flag=None, use_bytecode=True, vf_merge_level=1, warp_size=32)",
        "wrapper_depth": 0,
        "artifacts": [
          {
            "kind": "source",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.source",
            "sha256": "ad09da66a0244e59e3344a798e7c0f9d81e4b5a19430c9b4bdec74a03b8585a3"
          },
          {
            "kind": "ttir",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.ttir",
            "sha256": "f626d1b46f192b4125294981e5cc71f714326e98d4a4901f0b1f22883d956981"
          },
          {
            "kind": "ttadapter",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.ttadapter",
            "sha256": "fb70cd58eef1f3ae4e020936304234fc08e66dd5214274da00c4963f15443c78"
          },
          {
            "kind": "mlirbc",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.mlirbc",
            "sha256": "1961d1abc48643aff88e5122cd5d3bbba83f4784e5e40ffdc005aa7a9fa726e6"
          },
          {
            "kind": "bcmlir",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.bcmlir",
            "sha256": "2f12911b9647c6a5da0d340ea383018bc1d74cca1c923b1e59fda8a50cd24bc7"
          },
          {
            "kind": "npubin",
            "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/candidate/ir/_gn_kernel-0.npubin",
            "sha256": "f44b02a0bfde0d777ec741a61c2b5b71be3d7c623af8a94d208e14ab23941c20"
          }
        ]
      }
    ]
  },
  "source": {
    "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/starting-points/native_group_norm.py",
    "sha256": "12d25c47d259a2dad3ef596cd631d118fc649fd68e15333dd32b28529e672baa",
    "text": "# Copyright 2026, The FlagOS Contributors.\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n#\n# Generated by KernelGen: https://github.com/flagos-ai/KernelGen\n\"\"\"Native group normalization as a single fused Triton kernel.\n\nOne program per (group, batch) pair.  Each program\n\n1. walks its whole group (CG channels x HW elements) accumulating sum and\n   sum-of-squares in fp32 with a chunked per-channel loop,\n2. derives ``mean`` / ``rstd`` and stores them,\n3. re-walks the same range applying ``y = x * a + b`` with per-channel\n   ``a = rstd * w`` and ``b = bias - mean * rstd * w``.\n\nMeasured on this target, the whole operator is host-overhead bound (the\nreference operator itself measures ~30us/call for every shape), so the wrapper\nlaunch one kernel, does one output allocation plus one fused mean/rstd\nallocation, and re-uses the compiled kernel's own launch entry once Triton has\nbuilt it for the current specialization.\n\"\"\"\nimport torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\ndef _gn_kernel(\n    X,\n    OUT,\n    MEAN,\n    RSTD,\n    W,\n    B,\n    eps,\n    HW: tl.constexpr,\n    CG: tl.constexpr,\n    G: tl.constexpr,\n    NS: tl.constexpr,\n    CS: tl.constexpr,\n    SS: tl.constexpr,\n    COUNT: tl.constexpr,\n    BLOCK: tl.constexpr,\n    HAS_AFFINE: tl.constexpr,\n    EVEN: tl.constexpr,\n):\n    g = tl.program_id(0)\n    n = tl.program_id(1)\n    c0 = g * CG\n    xb = X + n * NS + c0 * CS\n    idx = tl.arange(0, BLOCK)\n\n    accs = tl.zeros([BLOCK], dtype=tl.float32)\n    accq = tl.zeros([BLOCK], dtype=tl.float32)\n    for ci in range(0, CG):\n        co = ci * CS\n        for start in range(0, HW, BLOCK):\n            hi = start + idx\n            if EVEN:\n                xf = tl.load(xb + co + hi * SS).to(tl.float32)\n            else:\n                msk = hi < HW\n                xf = tl.load(xb + co + hi * SS, mask=msk, other=0.0).to(tl.float32)\n            accs += xf\n            accq += xf * xf\n\n    inv = 1.0 / COUNT\n    mean = tl.sum(accs, 0) * inv\n    var = tl.sum(accq, 0) * inv - mean * mean\n    rstd = 1.0 / tl.sqrt(var + eps)\n\n    midx = n * G + g\n    tl.store(MEAN + midx, mean.to(MEAN.dtype.element_ty))\n    tl.store(RSTD + midx, rstd.to(RSTD.dtype.element_ty))\n\n    # OUT is allocated contiguous as (N, C, HW): group (n, g) covers\n    # [midx * COUNT, (midx + 1) * COUNT) and element (ci, hw) sits at\n    # ci * HW + hw.  The input side keeps its own strides.\n    cb = midx * COUNT\n    for ci in range(0, CG):\n        co = ci * CS\n        oo = ci * HW\n        if HAS_AFFINE:\n            a = rstd * tl.load(W + c0 + ci).to(tl.float32)\n            b2 = tl.load(B + c0 + ci).to(tl.float32) - mean * a\n        else:\n            a = rstd\n            b2 = -mean * rstd\n        for start in range(0, HW, BLOCK):\n            hi = start + idx\n            if EVEN:\n                xf = tl.load(xb + co + hi * SS).to(tl.float32)\n                tl.store(OUT + cb + oo + hi, (xf * a + b2).to(OUT.dtype.element_ty))\n            else:\n                msk = hi < HW\n                xf = tl.load(xb + co + hi * SS, mask=msk, other=0.0).to(tl.float32)\n                tl.store(\n                    OUT + cb + oo + hi,\n                    (xf * a + b2).to(OUT.dtype.element_ty),\n                    mask=msk,\n                )\n\n\n_BLOCK_CACHE = {}\n_CK_CACHE = {}\n_STREAM_FN = None\n\n\ndef _as_int(value):\n    if isinstance(value, int):\n        return value\n    try:\n        return int(value)\n    except Exception:\n        return int(value.value)\n\n\ndef _stream_slow():\n    return _as_int(torch.npu.current_stream()._as_parameter_)\n\n\ndef _stream_candidates():\n    cands = []\n    try:\n        import torch_npu\n\n        mod = getattr(torch_npu, \"_C\", None)\n        fn = getattr(mod, \"_npu_getCurrentRawStream\", None)\n        if fn is not None:\n            cands.append(lambda fn=fn: _as_int(fn(0)))\n        fn = getattr(mod, \"_npu_getCurrentStream\", None)\n        if fn is not None:\n            cands.append(lambda fn=fn: _as_int(fn(0)._as_parameter_))\n    except Exception:\n        pass\n    return cands\n\n\ndef _resolve_stream():\n    \"\"\"Cache a cheap callable that returns this device's current stream handle.\n\n    Only a candidate that reproduces the full ``current_stream()`` handle is\n    accepted, so the fast path can never pick a different stream than the one\n    the caller is using.\n    \"\"\"\n    global _STREAM_FN\n    if _STREAM_FN is not None:\n        return\n    try:\n        good = _stream_slow()\n    except Exception:\n        return\n    for cand in _stream_candidates():\n        try:\n            if cand() == good:\n                _STREAM_FN = cand\n                return\n        except Exception:\n            continue\n\n\ndef _stream_ptr():\n    fn = _STREAM_FN\n    if fn is not None:\n        try:\n            return fn()\n        except Exception:\n            pass\n    return _stream_slow()\n\n\ndef _pick_block(hw):\n    blk = _BLOCK_CACHE.get(hw)\n    if blk is None:\n        blk = 64\n        while blk < hw and blk < 4096:\n            blk <<= 1\n        _BLOCK_CACHE[hw] = blk\n    return blk\n\n\ndef _cache_keys():\n    keys = set()\n    caches = getattr(_gn_kernel, \"device_caches\", None)\n    if not caches:\n        return keys\n    for entry in caches.values():\n        cache = entry[0] if isinstance(entry, (tuple, list)) else entry\n        try:\n            keys.update(cache.keys())\n        except Exception:\n            pass\n    return keys\n\n\ndef _find_new(before):\n    caches = getattr(_gn_kernel, \"device_caches\", None)\n    if not caches:\n        return None\n    for entry in caches.values():\n        cache = entry[0] if isinstance(entry, (tuple, list)) else entry\n        try:\n            for key, value in cache.items():\n                if key not in before:\n                    return value\n        except Exception:\n            pass\n    return None\n\n\ndef run(input, weight, bias, N, C, HxW, group, eps):\n    x = input\n    dt = x.dtype\n    device = x.device\n    out = torch.empty(x.shape, dtype=dt, device=device)\n    buf = torch.empty((2, N, group), dtype=dt, device=device)\n    mean = buf[0]\n    rstd = buf[1]\n\n    dim = x.dim()\n    st = x.stride()\n    ns = st[0]\n    cs = st[1]\n    if dim > 2:\n        ss = st[dim - 1]\n        if dim > 3:\n            acc = 1\n            shape = x.shape\n            for d in range(dim - 1, 2, -1):\n                if st[d] != ss * acc:\n                    x = x.contiguous()\n                    st = x.stride()\n                    ns = st[0]\n                    cs = st[1]\n                    ss = st[dim - 1]\n                    break\n                acc *= shape[d]\n    else:\n        ss = 1\n\n    hw = HxW\n    cg = C // group\n    count = cg * hw\n    block = _pick_block(hw)\n    even = hw % block == 0\n    has_affine = weight is not None and bias is not None\n    wsrc = x if weight is None else weight\n    bsrc = x if bias is None else bias\n\n    xp = x.data_ptr()\n    op = out.data_ptr()\n    mp = mean.data_ptr()\n    rp = rstd.data_ptr()\n    wp = wsrc.data_ptr()\n    bp = bsrc.data_ptr()\n\n    key = (\n        dt,\n        wsrc.dtype,\n        bsrc.dtype,\n        has_affine,\n        xp & 15,\n        op & 15,\n        mp & 15,\n        rp & 15,\n        wp & 15,\n        bp & 15,\n        hw,\n        cg,\n        group,\n        ns,\n        cs,\n        ss,\n        count,\n        eps,\n        block,\n        even,\n    )\n    ent = _CK_CACHE.get(key)\n    if ent is not None and ent is not False:\n        ent[0](\n            group,\n            N,\n            1,\n            _stream_ptr(),\n            ent[1],\n            ent[2],\n            None,\n            None,\n            None,\n            xp,\n            op,\n            mp,\n            rp,\n            wp,\n            bp,\n            eps,\n            hw,\n            cg,\n            group,\n            ns,\n            cs,\n            ss,\n            count,\n            block,\n            int(has_affine),\n            int(even),\n        )\n        return out, mean, rstd\n\n    if ent is False:\n        _gn_kernel[(group, N)](\n            x,\n            out,\n            mean,\n            rstd,\n            wsrc,\n            bsrc,\n            eps,\n            HW=hw,\n            CG=cg,\n            G=group,\n            NS=ns,\n            CS=cs,\n            SS=ss,\n            COUNT=count,\n            BLOCK=block,\n            HAS_AFFINE=has_affine,\n            EVEN=even,\n        )\n        return out, mean, rstd\n\n    before = _cache_keys()\n    try:\n        _gn_kernel[(group, N)](\n            x,\n            out,\n            mean,\n            rstd,\n            wsrc,\n            bsrc,\n            eps,\n            HW=hw,\n            CG=cg,\n            G=group,\n            NS=ns,\n            CS=cs,\n            SS=ss,\n            COUNT=count,\n            BLOCK=block,\n            HAS_AFFINE=has_affine,\n            EVEN=even,\n        )\n        ck = _find_new(before)\n        if ck is not None and hasattr(ck, \"run\"):\n            _CK_CACHE[key] = (\n                ck.run.launch,\n                ck.function,\n                ck.packed_metadata,\n            )\n            _resolve_stream()\n        else:\n            _CK_CACHE[key] = False\n    except Exception:\n        _CK_CACHE[key] = False\n    return out, mean, rstd\n\n\nnative_group_norm = run\n",
    "full_characters": 9641,
    "truncated": false,
    "read_full_on_demand": false
  },
  "provenance": {
    "capture_state": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/analysis/native_group_norm/paired/state.json",
    "sha256": "ecfb49c3c2b48424d49a6a1f10a32438c63d970e6e4d8472f797077db2b806f5"
  },
  "bounded_historical_prior": [
    {
      "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/skills/hgkda/references/算子类型/归约与扫描.md",
      "sha256": "bde5d4b2cc1478ad8a638d9be210e4779bcef1cc8478cf691f220c06ee8ded1d",
      "text": "# 归约、扫描与随机数\n\n归约关注维度、归约长度、输出覆盖和中间精度，检查是否先转置或写出临时结果。min 可评估输入精度比较；sum、量化 scale 和扫描不可照搬 min 的精度策略。\n\n扫描还包含块内扫描、块间前缀传播与最终修正。复杂扫描先画出实际阶段，区分计算、阶段间依赖和临时存储；metrics 无法定位时追加指令或热点采集。\n\n## 梯形累计积分：精度不仅是累加dtype\n\n`cumulative_trapezoid(y, dx=dx)`的数学式可以融合，但参考实现的舍入位置也是数值语义的一部分。endurance-20261022的同输入NPU任务记录显示参考依次执行Slice、Add、Muls、Cumsum；固定FlagGems在kernel内直接将两侧输入扩为FP32相加、扫描，最后另乘dx，在4097长度的FP16/BF16补充检查中超差。\n\n本批CPU表达式探针对三种dtype、四种dx比较：`cumsum((left + right) * (dx / 2))`逐元素与CPU参考相同；把相邻元素先升FP32相加、或把非二次幂dx移到扫描之后会产生差异。原始证据为该批`research/cumulative-rounding-cpu.json`，CPU结论不能替代NPU验证。\n\n融合候选首先需要保留低精度Add和缩放的舍入；扫描的输入dtype、实际累加精度及结合顺序还要分别核对，不能预设升FP32扫描就能对齐参考。非二次幂dx、长序列、转置和非末轴都要额外验证。\n\n第二批实际KG探针发现了CPU建议的限制：在NPU的4×4097输入及六种dx上，FP16增量直接调用`cumsum`与`cumulative_trapezoid`逐元素一致；先将相同增量转FP32再`cumsum`、最后转回FP16，最大误差为0.01953–0.375。该探针的BF16/FP32对应路径则对齐。原始探针见`research/cumulative_trapezoid/reference-cumsum-dtype-probe/`。其中`f16acc`只是探针字段名，实际表达式为`inc.cumsum`，不能据此断言硬件累加器一定是FP16；输入dtype可能改变算法、结合顺序或中间精度，需要实际IR或指令再区分。早期“前两次舍入＋FP32扫描即可”的说法因此仅保留为已受反例限制的候选思路。\n\nCPU建议在第一批cumulative_trapezoid角色冻结后形成，NPU反例来自第二批模型自行提交的真实探针；知识修订供之后新覆盖层使用，不改写已冻结角色或原始结果。\n\n扫描按形状切换实现时，必须额外在性能输入上检查正确性。Native合同的`correctness`与`timing`是两个phase，前者小形状通过不能替代后者大形状的数值验证；本批独立门禁新增全部9项冻结性能输入，各两次及切stream，仍使用原容差。矩阵乘前缀方案还应核对三角矩阵方向、块间carry、最后不足整块时的mask和FP32 dot精度。局部dot很快或整组计时分数很高，都不能替代这些检查。\n\nuniform 属于随机数生成。验证状态推进、种子、区间、dtype 与统计性质；不能把“不能逐值对齐不同 RNG 实现”当作任意结果都正确，也不能为加速重复固定输出。性能比较须保持原测试合同。\n\n替换Philox为计数器哈希时，须单独报告算法和统计限制。输出均值、方差及连续两次不同，不能排除seed/offset被截断：endurance-20261022候选曾把两者掩为31位，CPU表达式见证表明第31/32/40位不同的状态会别名。额外门禁因此检查高位seed、高位offset、相同状态复现、状态推进与恢复重放；不以此声称通过完整随机质量测试。状态递增单位也需说明，不能只因调用同一状态获取函数就声称与原Philox逐项消耗相同。\n\nper_token_quant_int8 包含逐行绝对值最大值、scale 和量化写回，分别检查归约与逐元素阶段；候选性能需包括全部输出和 scale。\n\n## MoE 归约：分开输入规模和执行路径\n\n`moe_sum_reduce` 的输入为 `[token, top_k, hidden]`，沿 `top_k` 求和后乘缩放系数。保持FP32累加、输出dtype、全部输入stride及非默认scale。把更多token合并到一个program时，同时核对中间张量规模、尾部覆盖和实际grid，不能只按tile越大越好选择。\n\n`night-20261021-v3` 中，固定原版的大输入配置为1×2048；无profiler第二轮采用16×1024。4096×8×7168的bf16输入，grid从16384个program减少到1792个，两次完整复验相对固定原版为2.876× / 2.904×；小输入保留原配置，仍慢于PyTorch。该结果支持按规模分流，不证明所有输入都受HBM带宽限制。\n\n同批原生profiler组采用连续且整除时的快路径、编译期top-k循环及更少stride传参，保留通用分支。其8项合同两次复验相对PyTorch为1.543× / 1.649×，36项扩展语义也全部通过。具体tile值是本版本、本合同的测量结果，不作为其他dtype、布局或芯片的默认最优配置。该轮修改早于最终profiling，不能写成profiling建议产生的收益。\n\n来源：该批 `results/moe_sum_reduce/`；无profiler源码SHA `69caad9f45e9ebc794f0561a62a6b7838f45836962aa9d14c5cedbfdeb14fe97`，原生profiler源码SHA `96b0e8e40b23afe9a654d9e1afdb120046c97edebeccea5d998c212fa8491196`。\n\n## 同一分块在一般布局上也需重新核对编译资源\n\n上述无profiler版扩展语义为30/36。六个失败是stride2的 `[129,2,257]` 和 `[129,16,1025]`，覆盖三种dtype；原配置在大token时仍采用16行分块。FP16/BF16编译报告分别要求6291456和12582912 bits UB，而可用1572864 bits；FP32同两形状分别要求4194304和8388608 bits。原始失败回执SHA为 `79ddf2fc170600989953dfed80679b705867e54f9f312a17f321c98a81ce734a`。\n\n这说明不能把连续、对齐输入的可编译tile直接推广到一般stride，也不能用元素数乘dtype字节数代替全部编译缓冲需求。建议保留已验证的连续对齐快路径，一般布局先限制每program处理量，核对实际编译资源及完整语义；本批追加的单行、最多256列候选仍待NPU验证。\n\n原HG-KDA版扩展语义31/36，失败涉及奇数行宽下的数值或输入保持检查。其配置将奇数hidden的列块缩为1，`[65,8,1025]` 和 `[129,16,1025]` 分别产生66625和132225个program。program数是源码几何事实，不是已证明的失败原因，也不能据此直接宣称超过某个硬件启动上限。另有带mask列块的追加候选；原始失败及具体编译/执行机制尚未闭环前，不把正式8项通过扩大为一般布局正确。\n",
      "origin": "历史经验与失败约束；不是本批实测发现"
    },
    {
      "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/skills/hgkda/references/性能指标/指令与同步.md",
      "sha256": "f071dbb34201d4281a247d3496c9cae52dbeee3190d687f3a36c6555cbea8f11",
      "text": "# 指令、同步与热点\n\n仅在 metrics 无法区分具体机制或存在多阶段依赖时升级采集。先查看 KGS 声明的 levels/capabilities；目前 Ascend 常见为 metrics 与 instruction，不能虚构独立 source 级别。\n\ninstruction 使用模拟器或后端指令产物。核对目标 kernel、源码、case 和映射覆盖后，检查搬运、类型转换、循环、SET_FLAG/WAIT_FLAG 等同步与热点位置。没有源码/PC 对应关系时，注明定位范围，不把附近指令当作该行实测证据。\n\n模拟器时间和指令计数服务于诊断，正式性能由同一完整 KGS 合同的真机评测给出。采集失败或能力缺失保留原因；不要以重放真实硬件故障的方式无限补测。\n\n## 流水比例不能还原时间线\n\nMoE本轮大输入的Vector约38.34%、MTE2约67.15%，加上Scalar和MTE3约为111.5%。这些边际活跃统计不能推出“只有11.5%重叠”或“89%串行”，也不能直接据此预测约150 μs可以被隐藏。需要观察指令/同步顺序，或实际修改循环与stages，再同时比较设备时长、正式性能和流水变化。\n\n`num_stages=2` 出现在源码中也不证明编译后实现了双缓冲。可以将循环形式、预取深度或tile作为可证伪实验，但须保留相同输入语义，检查UB/缓冲占用及实际生成路径；编译失败与性能回退分别记录。\n\n核对采样聚合规则。本批KGS的op_summary含10次预热＋100次测量，summary按算子排除前10条后聚合。MoE最小case的110条均值为1.9360 μs，排除前10条为1.78984 μs，与summary的1.790 μs一致。不能把CSV中存在预热记录解释为summary也包含预热。\n",
      "origin": "历史经验与失败约束；不是本批实测发现"
    }
  ]
}

本批起点KGS结果摘要（完整原件见结果路径）：
{"status": "PARTIAL_PASS", "geo_mean": null, "num_passed": 21, "num_workloads": 27, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmpsh135g4g/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmpsh135g4g/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/GHKFQCFKBKFHWANMILFPP7AXSV7CYUO4DEII4EOVM4TT6SWYLDWQ\n\n///------------------[ERROR][Triton][END]------------------\n\n", "cases": [{"uuid": "native_group_norm-timing-0", "phase": "timing", "status": "PASSED", "latency_ms": 0.16862540596124684, "reference_latency_ms": 0.034894265923079336, "log": ""}, {"uuid": "native_group_norm-timing-1", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmpsh135g4g/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmpsh135g4g/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/GHKFQCFKBKFHWANMILFPP7AXSV7CYUO4DEII4EOVM4TT6SWYLDWQ\n\n///------------------[ERROR][Triton][END]------------------\n\n"}, {"uuid": "native_group_norm-timing-2", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmpbab_hotz/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmpbab_hotz/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/REL7IAJBOG6K5TQTN2WCGH2MXCAE3VGLCY6GZY6KHY6QOLZJ5LNA\n\n///------------------[ERROR][Triton][END]------------------\n\n"}, {"uuid": "native_group_norm-timing-3", "phase": "timing", "status": "PASSED", "latency_ms": 0.1701250769747296, "reference_latency_ms": 0.035954101619405376, "log": ""}, {"uuid": "native_group_norm-timing-4", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmpch_0hofr/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmpch_0hofr/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4718592 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/XJEIGAHV6E3BM2NYVPRNIGXD7ABZMWHBQX5UGQFW4U35T7YVT43Q\n\n///------------------[ERROR][Triton][END]------------------\n\n"}, {"uuid": "native_group_norm-timing-5", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmp45sy5gib/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmp45sy5gib/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2359296 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/EJCV2QAMEXJG4PSTOF256KPVYFENKI2FA7UD6TPLWTRLX5NFQTRQ\n\n///------------------[ERROR][Triton][END]------------------\n\n"}, {"uuid": "native_group_norm-timing-6", "phase": "timing", "status": "PASSED", "latency_ms": 0.16747669820391098, "reference_latency_ms": 0.034478637310646716, "log": ""}, {"uuid": "native_group_norm-timing-7", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmpgyjg4kyv/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmpgyjg4kyv/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4194304 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4194304 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4194304 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4194304 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 4194304 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/4QQLCO3SUOIKPK2VACI2IWQJJAVBHBWOMMFLISKNFO7OD2VVNI4Q\n\n///------------------[ERROR][Triton][END]------------------\n\n"}, {"uuid": "native_group_norm-timing-8", "phase": "timing", "status": "RUNTIME_ERROR", "latency_ms": null, "reference_latency_ms": null, "log": "Traceback (most recent call last):\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 330, in compile\n    next_module = compile_ir(module, metadata)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 1136, in <lambda>\n    stages[\"npubin\"] = (lambda src, metadata: _compile_linalg_to_npu_bin(src, metadata, options))\n                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 659, in _compile_linalg_to_npu_bin\n    return linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/compiler.py\", line 841, in linalg_to_bin_enable_npu_compile_A2_A3\n    ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/subprocess.py\", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command '['/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile', '/tmp/kernelgen-isolated-dgecdp_j/tmphkc4uggy/kernel.mlir', '--target=Ascend910B4-1', '--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True', '--link-aicore-bitcode=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/kernelgen-isolated-dgecdp_j/tmphkc4uggy/kernel']' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 628, in evaluate\n    reference_latency, candidate_latency = self._timing_trial(\n                                           ^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 519, in _timing_trial\n    candidate_invoke()\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/kernelgen_server/evaluation/engine.py\", line 515, in <lambda>\n    lambda call=candidate_call: candidate(*call.args, **call.kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/root/.cache/kernelgen_server/triton/f308ffb4956b4df6bb16-1_o7jife/source/main.py\", line 26, in native_group_norm\n    return group_norm(input, weight, bias, N, C, HxW, group, eps)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\", line 212, in group_norm\n    group_norm_kernel[grid](\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 420, in <lambda>\n    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/libentry.py\", line 1535, in run\n    kernel = self.fn.run(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 736, in run\n    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py\", line 864, in _do_compile\n    kernel = self.compile(src, target=target, options=options.__dict__)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/experimental/tle/__init__.py\", line 85, in tle_compile\n    compiled_kernel = original_compile_fn(src, target, options)\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py\", line 351, in compile\n    raise MLIRCompilationError(stage_name, error_detail) from e\ntriton.compiler.errors.MLIRCompilationError: \n///------------------[ERROR][Triton][BEG]------------------\n[ConvertLinalgIRToBinary] encounters error:\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2097152 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2097152 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2097152 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2097152 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/ops/groupnorm.py\":31:0): error: ub overflow, requires 2097152 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /root/.triton/cache/V56CYDKE5OIA5KWNPFRB4DQHRL42RNAUKILNLTCDG4IJ2A32PFIA\n\n///------------------[ERROR][Triton][END]------------------\n\n"}]}
