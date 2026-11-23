# 夜间新增IR案例与GQA入口

昇腾统计表已完成，两个不同算子的固定源码IR案例已验收。完整合同与既有补测保持原样，图中隐藏同实现测量，实际测量与父链继续保留。

| 案例 | 目标输入 | 本轮原IR → 新IR | 完整验收 | 全合同候选耗时几何变化 |
| --- | --- | --- | --- | --- |
| native_group_norm | FP32 `[8,64,64,64]`，groups=8，timing-7 | 24.812188 → 23.734423μs，下降4.34% | 27/27、双方42/42、条件交错与后重放 | 下降约0.085% |
| value_selecting_reduction_backward | FP32 output `[8192,8192]`，dim=1，keepdim=false，timing-14 | 187.771289 → 183.792334μs，下降2.12% | 96/96、双方96/96、条件交错与后重放 | 上升约0.26% |

目标收益由同合同正式结果与交错检查支持；完整集合变化同时记录。历史跨时间计时与参考变化归入详情，主图逐case采用共同CANN锚点重算。原始交付成绩与源码身份冻结，图中历史重算值单独标识。

GroupNorm通过保留统计首遍8通道输入，减少第二遍GM读取；设备指标确认256→128KiB/program，MTE2服务4.489→2.488μs，Scalar等待2.413→3.298μs。后模拟器时钟核验41/44个PC匹配，原版21/21；两个窗口分别归一化，部分窗口、退出与3个局部差异保存。物理UB峰值和spill仍需专门观测。

回填IR先写整行零再覆盖有效位置，dense broadcast2→0、compare1→0、select1→0；Vector服务6.866→0.325μs/program。MTE3指令17→33、输出512→约514KiB/program，MTE3服务和Scalar等待增加。后窗口MTE3 productive约96.73%，时钟10/12个PC匹配；原版26/29。应用未完成，部分窗口分别解释。ir-001语法失败后由模型修复为ir-002，真实IR修复父链单列，图上仅显示完整合格节点。

- GroupNorm：[完整验收](native-group-norm-hypothesis/selection.json)、[主图](../native_group_norm-simulator-20261120/index.html)、[固定输入](../native_group_norm-shape-20261120/index.html)、[重放事实](native-group-norm-hypothesis/post-mechanism-ir002-facts.json)。
- 回填：[完整验收](value-selecting-hypothesis/selection.json)、[主图](../value_selecting_reduction_backward-simulator-20261120/index.html)、[固定输入](../value_selecting_reduction_backward-shape-20261120/index.html)、[修复父链](value-selecting-hypothesis/retention-lineage-ir002.json)、[重放事实](value-selecting-hypothesis/post-mechanism-ir002-facts.json)。
- [26项总览](../operator-traces-stitched-20261117/index.html)、[26页夜间PPT](../group-meeting-20261119/whitebox-simulator-ascend-20261120-night.pptx)、[图验收](trace-validation.json)、[PPT原生渲染](native-slides/slide-25.png)。
- [昇腾统计调研](../ascend-statistics-20261120/README.md)。

GQA 已验收新最佳：固定官方 case1/2/4/7 全部通过，目标 BF16 case4 从历史 **1378.73655→784.63765μs，下降43.09%**；四项候选耗时几何平均下降15.17%。本次改动为Triton宿主分块选择：非因果KV 64→128，因果保持64；IR用于机制解释。修改前双分析读取与决策、实际源码父链、首版UB超限及修复、修改后双分析和目标二进制绑定均已保存。后重放三核时钟19/19、43/43、40/40通过，应用未完整结束，部分窗口分别解释；完整20项官方验收继续列为后续工作。

- GQA：[完整案例](gqa/README.md)、[连续最佳曲线](../gqa-simulator-20261120/index.html)、[保留决定](gqa/selection.json)、[四项原始结果](gqa/iterations/attempt-002/parent-validation/results.json)。
- 最终汇报：[28页PPT](../group-meeting-20261119/whitebox-simulator-ascend-20261120-night-final.pptx)、[新增曲线页](native-slides/slide-27.png)、[机制页](native-slides/slide-28.png)、[PPT验收](completion-slides.json)。原24页与26页版本继续保留。
- 回填ir-003完整96/96、双方96/96、后重放13/14个PC匹配；184.796923μs慢于保留183.792334μs，作为[完整负结果](value-selecting-hypothesis/retention-ir003-negative.json)归档。量化cachemask双分析保留为观察池。
- 模型已只读回收本轮完整结果；[原始回复](model-result-review/response.md)、[宿主纠正](model-result-review-host-note.md)共同保存。

三阶段均已完成。KGS 19730保持运行；本轮设备评测、后重放及模型调用均已收尾。实际IR与Triton候选继续由CC／DeepSeek编写，Codex负责分析、基础设施、评测与审查。

[三阶段完成核对](completion-audit.json)。
