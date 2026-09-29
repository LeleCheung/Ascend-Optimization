# Roofline 分析




交付的 `roofline.py` 工具接受两个输入文件：共享配置 `manifest.json` 与逐用例数据 `points.jsonl`。





## 1\. 安装与运行





```PowerShell
python -m pip install matplotlib

python roofline.py manifest.json points.jsonl output --plot
```



`output` 必须不存在或为空，防止旧结果混入新实验。不需要图片时去掉 `--plot`。



## 2\. manifest\.json



保存整套实验共享的设备、后端、基准测试、性能分析工具和性能上限。每个数据类型只有一个标准计算性能上限，整套实验共用一个 HBM/DRAM 带宽上限。

```JSON
{
  "schema_version": "1.0",
  "dataset_id": "h20_flaggems_69cd0b8",
  "platform": {
    "vendor": "nvidia",
    "device": "NVIDIA H20-3e",
    "implementation": "FlagGems"
  },
  "benchmark": {
    "name": "FlagGems benchmark",
    "commit": "69cd0b8dcdb014ffc674a5818b33b6c2c68ea1e8"
  },
  "profiler": {"name": "NCU", "version": "2026.1.0"},
  "compute_roofs": {
    "float32": {"compute_path": "cuda_core", "tflops": 60.0, "source": "empirical"},
    "float16": {"compute_path": "tensor_core", "tflops": 200.0, "source": "empirical"},
    "bfloat16": {"compute_path": "tensor_core", "tflops": 200.0, "source": "empirical"}
  },
  "bandwidth_roof": {"memory_level": "DRAM", "gbps": 4299.83, "source": "empirical"}
}
```



必填：`platform.vendor/device/implementation`、所测精度对应的正数 `compute_roofs.<dtype>.tflops`，以及正数 `bandwidth_roof.gbps`。



## 3\. points\.jsonl



采用 UTF-8 编码的 JSONL，每个非空行对应一组“用例 × 数据类型”。



```JSON
{"case_id":"bf16_mm_001","op":"mm","nodeid":"benchmark/test_mm.py::test_mm","dtype":"bfloat16","status":"passed","metrics":{"flops":3221790720,"bytes":44567040,"duration_us":25.4,"scope":"kernel-sum","kernel_count":1},"sources":{"flops":"hardware_counter","bytes":"hardware_counter","time":"device_counter","source_path":"raw/bf16_mm_001.ncu-rep"},"capture":{"interval_match":true,"duration_ratio":1.0},"kernels":[{"name":"mm_kernel","duration_us":25.4}],"metadata":{}}
```



成功用例必填：`case_id`、`op`、`dtype`、`status`、`metrics.flops/bytes/duration_us`。单位依次为浮点运算次数、字节、微秒。



失败、超时或不支持的用例也可保留。



```JSON
{"case_id":"bf16_softmax_001","op":"softmax","dtype":"bfloat16","status":"timeout","metadata":{"error":"case exceeded 300 seconds"}}
```



关键口径：浮点运算次数、字节数和时间必须覆盖同一个内核或采集范围；融合乘加通常计为两次浮点运算。优先采用硬件计数；算法估算的浮点运算次数或逻辑字节数必须在 `sources` 中如实标记。



## 4\. 输出



```Plain Text
output/
├── all_points.csv
├── summary.json
├── fp32/
│   ├── roofline.png
│   ├── knee_analysis.csv
│   ├── operators/<op>/point.json + roofline.png
│   └── rejected/<case>/point.json
├── fp16/
└── bf16/
```



- `all_points.csv`：全部用例、有效性、拒绝原因和派生指标。

- `summary.json`：程序读取的总量、精度和质量汇总。

- `<dtype>/knee_analysis.csv`：有效点的拐点分类、TFLOP/s、上界和达成率。

- `<dtype>/roofline.png`：精度聚合图。

- `<dtype>/operators/<op>/`：单算子数据和图片。

- `<dtype>/rejected/`：失败或无效记录

    

## 5\. 统一公式



```Plain Text
AI                    = FLOPs / Bytes
Performance (TFLOP/s) = FLOPs / duration_us / 10^6
Bandwidth (GB/s)      = Bytes / duration_us / 10^3
Knee AI               = Compute Roof × 1000 / Bandwidth Roof
Roofline Bound        = min(Compute Roof, AI × Bandwidth Roof / 1000)
Efficiency            = Performance / Roofline Bound
```

