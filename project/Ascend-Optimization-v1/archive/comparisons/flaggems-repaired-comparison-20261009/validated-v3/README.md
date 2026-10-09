# 补偿归约的三轮完整验证

PReLU 补偿归约版基线及候选三轮均通过 21 项正确性、9 项计时。直接加速比中位数为 5.79×，但基线相对 PyTorch 仅 0.49×，主要因为补偿归约引入额外开销，因此不作为最终周报的 PReLU 性能对照。

此目录保留完整校验和逐 case 数据，用于证明归约精度问题可修复。BatchNorm 和 SmoothL1 的数据与最终批次相同，直接加速比分别为 2.19×、1.61×。

最终采用的基线与结果见 [总入口](../README.md)。本批次结果见 [comparison.json](comparison.json)、[per-case.csv](per-case.csv)，源码和完整请求见 [measurements/](measurements/)。
