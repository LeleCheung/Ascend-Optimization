# FlagGems 原生验收接入：log10_

2026-11-11：新增外部固定版本 FlagGems 原生验收入口。同一接入源码完成 KGS、HG-KDA 补测与 FlagGems 目标文件测试。主线仍为 26/30，不新增交付数量。

## 接口与环境

公共命令为 python -m hgkda.runtime upstream-test --config /path/upstream-test.json。runtime/upstream_test.py 核对外部 commit、允许的源文件改动、源码 SHA、严格 Triton 审查、实际导入路径及设备锁；upstream_driver.py 支持明确指定未修改的上游文件；adapters/flaggems.py 汇总真实测试结果。进程返回零但内部测试失败、收集错误、没有用例或有效计时均不能判为通过。

固定 FlagGems commit：d6a8eec473517a3d68157b208eb9c057eb1d4c50。环境为 Ascend 910B4、CANN 9.0.0、Triton 3.5.1、FlagTree 0.6.0+ascend.gitc286cba6；只使用物理卡 7 与项目设备锁。源码版本以实际加载路径和 SHA 为准，不能用旧安装分发元数据替代。

## 结果

接入源码 SHA256：8a2d3e28b3390153ac785fb28ddc185d160de7aaff41284e4c2a4354ad60588d。

| 验证 | 结果 |
| --- | --- |
| KGS 完整合同 | 36/36；相对 CANN 0.855809×，accepted=true |
| PyTorch／CANN 补测 | 各 66/66 |
| 特殊值专项 | 15/15 |
| profiler 生命周期 | 12/12：普通调用、无关采集之后、候选采集中、采集结束后，覆盖三种 dtype |
| FlagGems 原生正确性 | 18/18，零跳过，与原调度器所选用例集合一致 |
| FlagGems 原生性能 | 9 个有效 dtype／shape 测点 |
| 原生口径相对固定 FlagGems | 4.454178×，逐点实测延迟比的几何平均 |
| 原生口径相对 PyTorch | HG-KDA 0.653586×；原版独立测量 0.151582× |

原生 benchmark 和 KGS 的输入集合、参考、计时不同，分开解释。原版测量一次，最终候选仅一次完整 KGS；中间源码变化后重新验证，不重复同一源码择高分。原版原生正确性也通过 18/18，与历史不同合同的失败记录分开保存。

## 修复过程

1. 整数地址直接启动：原生正确性通过，profiler benchmark 因厂商启动器读取 tensor.size() 而失败。
2. CC 改为 Tensor 参数，并修复 KGS 要求 run 与 log10_ 为同一函数对象的接口冲突。完整正确性通过，但 KGS 退化到 0.750834×，未达标。
3. Codex 对照逐 case 延迟、启动器源码与实测 profiler 状态，把宿主参数合同和停止采集后的状态残留问题交给 CC。CC 保留普通地址路径，在需要形状元数据时传 Tensor，KGS 恢复到 0.855809×。

所有候选及其启动封装由 CC／DeepSeek 修改；Codex 负责诊断、工作流、检查器、只读审查及编排。不是从零生成算子，也不宣称模型独立发现全部优化。部分模型调用达到轮次上限，已写入源码经过独立审查和完整实测后采用，调用状态原样归档。

新增方法记录涵盖入口别名合同、普通与 profiler 启动参数差异、采集结束后兼容检查、禁止未知执行状态下重试原地计算。静态门禁只新增 profiler 活跃状态这一宿主元数据查询，未开放采集控制或其他后端计算。

## 范围与证据

原生 tools/run_tests.py --ops log10_ 在收集无关 tests/test_scaled_mm.py 时遇到 FP8 编译错误，且调度器退出零不能覆盖内部失败。本轮直接运行未修改的目标文件，核对用例集合一致；不声明全仓库调度器、CI 或代码评审通过，未创建 PR。

历史选用源码 d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049 和分数保留。新增生命周期检查只验证本轮源码，不追溯声明历史候选通过。KGS 结束后，外层汇总曾读取错误文件名；随后从实际 audit-final.json 汇总已完成结果，没有重跑评测或改写性能数据。

- [交付与复现](../../operators/log10_/upstream/20261111/README.md)
- [三层验收清单](../../operators/log10_/upstream/20261111/manifest.json)
- [逐点性能对比](../../operators/log10_/upstream/20261111/comparison.json)
- [完整实验索引](../experiments/upstream-log10-20261111/files.json)

软件测试 64 项、skill 格式验证通过；33,503 份冻结迁移原件身份核验通过。最终发布包与设备实测包的 Python 运行源码相同，补充了接口和方法说明。候选、原始结果和完整模型轨迹分别存放在 operators 与 development，公共运行不依赖后者。
