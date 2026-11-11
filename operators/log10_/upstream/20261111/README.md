# log10_：FlagGems 原生验收

本目录保存外部 FlagGems 接入源码、补丁与三层证据。固定上游 commit：d6a8eec473517a3d68157b208eb9c057eb1d4c50。接入源码 SHA256：8a2d3e28b3390153ac785fb28ddc185d160de7aaff41284e4c2a4354ad60588d。

| 验收层 | 本轮结果 |
| --- | --- |
| KGS 固定合同 | 36/36；相对 CANN 0.855809×，达到 0.8 验收线 |
| HG-KDA 补充检查 | PyTorch 与 CANN 参考各 66/66；特殊值 15/15；profiling 生命周期 12/12 |
| FlagGems 原生正确性 | 18/18；与原调度器所选用例集合一致 |
| FlagGems 原生 core benchmark | 9 个有效测点；相对固定 FlagGems 原版 4.454178×；本轮相对 PyTorch 0.653586× |

原生 benchmark 与 KGS 的输入集合、参考和计时方式不同，分别报告。相对 FlagGems 的结果按相同 dtype／shape 的实测延迟相除后取几何平均，见 [comparison.json](comparison.json)。不能把两个独立批次的相对 PyTorch 加速比直接相除。

## 复现

在独立外部工作区检出上述 commit，应用 [integration.patch](integration.patch)。补丁已在干净固定版本上通过 git apply --check，只增加昇腾实现与后端导出，不修改测试。

    git -C /path/to/FlagGems apply --check /path/to/integration.patch
    git -C /path/to/FlagGems apply /path/to/integration.patch
    python -m hgkda.runtime upstream-test --config /path/to/upstream-test.json

复制 [config.example.json](config.example.json)，填写工作区、Python、SDK、设备锁与新输出目录。实际导入身份见 [native/import.json](native/import.json)。KGS 入口见 [公共接口](../../../../hgkda/docs/runtime.md)：用本目录 implementation.py 和 source-review.json 设置 run 的候选与审查；将 checks/check_profiler_lifecycle.py 加入该 run 的专项检查，预期 12 项。不能沿用历史源码的成绩。

## 调整与范围

CC／DeepSeek 完成后端导出、run／log10_ 同一函数对象绑定与 profiler 启动参数兼容调整，Triton 数学计算保持原样。普通调用使用动态地址；采集状态或历史状态要求元数据时使用 Tensor；状态未知时采用保守路径。涉及内部 profiler 接口，跨版本需重验。

全仓库调度器被无关 FP8 测试收集错误阻断。本轮运行未修改的 tests/test_log10.py 与 benchmark/test_log10.py，并核对正确性用例集合一致；不声明全调度器、全仓库 CI 或 PR 已通过。

历史选用版本及分数保留，本轮不增加交付算子数量。详细过程见 [开发报告](../../../../development/reports/flaggems-upstream-log10-20261111.md)；公开运行不依赖开发档案。
