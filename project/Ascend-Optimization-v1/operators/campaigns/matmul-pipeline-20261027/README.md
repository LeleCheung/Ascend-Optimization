# 矩阵乘片内复用与流水优化 Goal

以上一批Roofline核验后保留的父版为起点，目标是取得真实设备性能改善，并将可验证机制固化到HG-KDA。父版包含历史KG／HG-KDA与Codex贡献，不是原版FlagGems。

## 最终结果

已采用相邻M输出块共享B的候选。相对本批父版实际设备加速 **1.040×**；相对PyTorch由父版 **0.862×** 变为 **0.895×**。两版各通过一次完整KGS **42/42**，另通过 **18＋24项**补充检查。冻结后的同模板四项对照为 **1.059×**，用于确认复用方向，不参与重选版。详见[精简汇报](精简汇报.md)及[逐case对比](正式逐case对比.csv)。

硬件采集的GM／L2／L1总流量基本不变，1024／2048 FP16的执行时间降低；IR中的bias相关copy／alloc各少一处。证据支持执行组织与复用方向有效，不将其表述为减少HBM搬运量。

## 文件入口

- validation/delivered.py：最终源码，SHA 0c35cfbc23d9656cd47fb4168395f26928e6fbf0a3fe662a613a0636b877fb9c。
- validation/comparison.json：15项正式同case延迟、选版与PyTorch分数。
- search-v2、search-v3、agent-review：有限搜索及真实CC＋DeepSeek调用；首次导入失败原件在search和runtime-records保留。
- parent-profile、candidate-profile、analysis：同输入真机采集、IR及数值Roofline、中文图表。
- matched-confirmation：同模板复用方向的四项确认、交错计时CSV及编译原件。
- method-snapshot：运行时12份方法原件；final-method-snapshot：最终方法和公共依赖，差异另列，不覆盖前者。
- evidence-audit.json：初步7,479项；evidence-audit-final.json、remote-evidence-audit.json：Windows／910B最终各7,556项独立重算；sync-verification.json：同步SHA；remote-closure-check.json：共享服务空闲与项目卡锁释放状态。

本批不commit／push，GitHub保持原提交。前轮模型调用冻结之后新增的知识只供后续批次，不回改提示或归因。

## 执行方案

1. 冻结父版、Roofline证据及本机编译参数；不重跑已失败的单大输出tile。
2. 保留多个小输出块，分别尝试A／B双向复用、2×2输出分解和有限流水配置，由CC＋DeepSeek根据真实证据选择追加候选。
3. 同输入交错设备计时筛选，完整通过三精度、布局、边界与stream补充检查，再各做一次KGS原合同42项评测。
4. 对入选候选与父版重新采集同输入硬件指标，检查片内流量和编译结构是否符合预测。
5. 交付更快源码，或保留父版并给出无收益的具体证据；更新中文知识、精简报告与逐文件同步核验。

仅使用物理卡7项目互斥锁和独立tmux；不改共享镜像或服务、不提交推送GitHub。理论逻辑请求、官方计数字节和经验参考分别报告，模型建议与Codex实际修改分别记账。
