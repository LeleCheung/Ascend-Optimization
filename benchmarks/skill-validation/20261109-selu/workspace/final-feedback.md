# SELU 反馈

## 结论
正式 KGS 36/36 正确，补充检查两侧各 42/42；相对 CANN 仅 0.210037，未达 0.8 线。旧交付独立入口为 1.039292，当前候选回退，不能替换旧交付。

## 延迟边界
小形状约 80–85us，CANN 约 13us；大形状约 84/152us，CANN 约 52–61us。端到端延迟只能证明变慢，不能证明原因；需分别验证宿主调度（入口解析、stream/设备处理、launch 开销）与设备计算（kernel 本体）。

## diagnosis 纠正
1. 逐 block 平均时间是行级统计，与全任务字节数口径、program 数不同，不能相除构造 Roofline。
2. 一个 program 载荷（2KiB/16KiB）不等于每条 DMA 指令搬运粒度；bytes_per_mte2 仅为参考量级。
3. 零值或缺失只能记为缺失，不能解释为低于采样分辨率或真实为零。
4. FP16 算术是否减少 extf/truncf，须以新候选 IR 核验，当前无证据。

## 已证实工程反馈
- 全局 1% 压缩阈值修正仅限子进程。
- Triton 引用外层常量必须用 tl.constexpr，否则 Ascend 报 NameError。
- 张量指针已含 storage_offset，顶层重复相加致二次偏移，偏移视图检查 3/9，改相对基址 0 后 9/9。
- 公共符号 selu_ 与入口冲突致 KGS 加载 BuildError，改私有 _selu_entry 后通过。
- 补充检查不能替代 KGS 正式入口验证。

## 假设
f16 快路径降低向量占用、去 mask、16KiB block、exp2 替代未验证，不写入已验证成功方法；后续需重采同输入 profiling 并分开测量宿主与设备耗时。
