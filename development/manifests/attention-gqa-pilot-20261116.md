# Level 4 GQA 起点合同

本轮选择官方`official-tasks 2.0.0 Beta / level4/gqa`，先验证官方case **1、2、4、7**：包含FP16／BF16、G4／G8／G12、因果prefill及不等长Q/KV。该集合是明确子集，不代表全部20项官方用例、原生CANNBench harness验收或排行榜成绩。

- 官方golden文件、cases.yaml和desc.md保留SHA；输入按官方shape、dtype及[-1,1]范围生成，固定seed=1729+case_id。
- 正确性使用官方golden对相同输入进行CPU FP64计算。完整报告MERE和MARE，严格应用说明中的阈值：FP16 `2^-10`、BF16 `2^-7`，要求MERE<阈值且MARE<10倍阈值。附加allclose只作诊断，不能代替官方公式。
- 性能参考优先`torch.ops.npu.npu_fused_infer_attention_score`，BSND、原生GQA头共享，不物化K/V到全部Q头；因果掩码在调用前准备。若API或精度不满足，单列失败，不以其他参考偷换原生身份。
- 两侧计时采用相同输入、5次warmup、3组各20次完整调用，组边界同步并取组平均的中位数。记录这是批量完整调用墙钟口径，不是KGS计时，也不与旧算子分数拼接。
- 候选从空源码起点由CC／DeepSeek生成，全部设备计算严格Triton；允许为精度优先建立较慢的正确起点。Codex只写评测驱动、合同和方法说明。
- 第一次生成起记录trace、修改前decision、模型日志、源码SHA、门禁及逐路径只读审查、正确性／性能结果。编译或精度失败反馈给原模型修复。
