# batch_norm_backward：历史归档

验收看板华为列 G159 为 4.3885×，已超过 0.8×。此前选择该对象不符合新筛选要求，现退出主线；保留实验作为复现与方法参考。

历史实验见 [阶段报告](reports/batch-norm-backward-20261007/README.md)。本目录是归档位置，原始阶段报告中的计划描述当时任务。

最新三轮对照见 [修复复测](../../comparisons/flaggems-repaired-comparison-20261009/validated-v5/README.md)：候选相对本机修复版 FlagGems 为 **2.19×**，相对 PyTorch 为 **5.01×**，双方每轮完整通过 30 项正确性与 15 项计时。

历史统一复测的候选相对 PyTorch 为 5.03×，对应设备侧 kernel 计时。完整版本、正确性与优化细节见 [联合对照](../../comparisons/flaggems-comparison-20261009/README.md)；[原版及候选源码](../../comparisons/flaggems-comparison-20261009/source-audit/batch_norm_backward/)和 [测量证据](../../comparisons/flaggems-comparison-20261009/measurements/batch_norm_backward/)仍随原联合批次保存。
