| FlagGems-SGLang算子（2项） | FlagGems-SGLang实测 | HG-KDA（完整调用） | 完整评测 | 关键优化 | 状态 | 结果更新日期 |
|---|---|---|---|---|---|---|
| `gelu_and_mul` | 0.068×；原版扩展FP32正确性未通过 | **2.271×** | 19/19；18项语义、15项性能输入数值、9项缓存／偏移／stream | 对齐本机PyTorch GELU参考公式，连续二维分块，简化索引与启动 | 已交付；原版未达标后优化 | 2026-10-23 |
| `moe_sum_reduce` | 0.965× | **1.624×** | **两次各8/8正式；36项扩展语义** | 合并多个token归约，连续整除快路径，修复非连续布局 | 已交付归档；原版已达标后进一步优化 | 2026-10-22 |

| 口径与来源 | 说明 |
|---|---|
| 范围 | 成果主表保留gelu_and_mul与moe_sum_reduce两项。per_token_quant_int8、rotary_embedding仅有原版体检，silu_and_mul_masked参考口径待核，三项不进入成果主表，原始体检证据归档保留。 |
| 计时与门槛 | KGS完整调用walltime，相对等价PyTorch参考；>0.8×为达标，>1×表示快于PyTorch。 |
| 原版状态 | gelu_and_mul原合同19/19但扩展FP32正确性未通过；moe_sum_reduce原版0.965×，属于已达标后的进一步优化。 |
| 固定版本 | FlagGems-SGLang提交938612058a981487e439da514df7d1371cf31e68；FlagTree 0.6.0+ascend.gitc286cba6；Triton 3.5.1；CANN 9.0。 |
| 复测与贡献 | moe_sum_reduce显示两次同条件完整有效结果的几何平均1.624×，各8/8正式及36项扩展；HG-KDA生成后由Codex修复布局。GELU等优化同样保留KG/HG-KDA、CC/DeepSeek及Codex实际贡献，不归为纯自动独立效果。 |
| 关联表 | [FlagGems Table](https://www.yuque.com/up3wk0/twwy86/vcqrbcy2hp7gds4g) |

