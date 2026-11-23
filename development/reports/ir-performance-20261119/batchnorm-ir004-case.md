# BatchNorm：固定Triton源码，加宽输出IR

本例是持续模拟器重放中的单算子IR修改：继续读取当前版本重放证据，模型固定Triton源码修改IR，编译新二进制，完成真机验收后再次模拟器重放。修改前决策与修改后观测分别保留，前后窗口范围按下文解释。

FP32 NCHW `[8,64,64,64]`、`timing-7`。原生FlagGems81.775047μs；已有源码最佳历史44.912918μs，本轮原IR42.955284μs；修改后39.261332μs、0.717359×CANN。该次正式测量相对原IR耗时下降8.60%，27/27与双方各36/36通过，全合同几何平均1.852180×CANN。

修改前模拟器保留统计通道局部窗口，提供MTE2／VECTOR同步频繁的线索；成对profiling给出整核差距，实际BCMLIR显示归一化通道每块搬运、广播和写回。此前加宽统计通道的候选CPU编译因UB需求超限而失败。Codex据此建议仅加宽逐元素输出通道，CC／DeepSeek读取证据、登记决策并编写实际IR。该过程包含模拟器、源码／IR及宿主诊断三类贡献。

实际改法：统计、Welford递推、归约与运行状态更新保持原字节，仅在空间维可整除256的守卫内，把归一化／写回分块从8×128扩大至8×256，原分支兜底。目标空间维4096，输出通道循环32→16，连续搬运段512B→1KB，逐元素FP32运算顺序保持一致。Triton源码、启动网格及编译选项保持固定。

实际输入／输出IR生成的[逐行差异](batchnorm/agent-calls/ir-004/actual-ir.diff)与模型原件分别保存，可用于组会展示改动区域。

新增[完整输入导出收据](batchnorm-accepted-ir004-complete-inputs-receipt.json)覆盖402份材料。9份修改前核心输入逐项匹配模型调用前保存的SHA清单，包括profiling、热点、流水、上下文、实际低层产物及源码；原393份导出及收据继续保留。

| 身份 | SHA256 |
| --- | --- |
| Triton源码 | `b8d4d4a2c07dabfc308e94412d0b5c219eb56ee4c7dc5358016e0eb2782b2f56` |
| 原TTIR | `fb0bdf3547e0086bbfdd12e29b683b303c98733943ed02a161ffcbfb329eb634` |
| 修改TTIR | `574eddcba7a8b920c211880d4332fb710465bcda4a944dc3f81ae74c066a17cc` |
| 原目标二进制 | `7a391932534402cabe5699810ba005beff6acd3f97918fea18a76b6eeeebfa98` |
| 修改目标二进制 | `41483d746ba67cd3456434ea7d176386cbe6e987f3ab56d360d1baf5935be5fe` |

实际编译收据确认覆盖生效。其余27个AST二进制逐一相同；全合同逐case候选耗时几何改善1.015144倍，其余用例的测量变化单独保留。ABBA四段每段三次，采用安装版KGS的L2清理计时，并逐段核对正式评测二进制：

| 段 | 实现 | 三次测量中位数μs |
| --- | --- | ---: |
| A1 | 原IR | 44.947934 |
| B1 | 修改IR | 40.058368 |
| B2 | 修改IR | 39.022345 |
| A2 | 原IR | 42.703993 |

修改后paired task39.880798μs。模拟器应用执行完成，33个PC时钟比较全部通过，绝对时间单位校验通过。当前覆盖二进制的直接PC源码映射缺失，保留PC热点及源码↔IR语义对应。修改前局部窗口与修改后完整窗口分别解释；本例验证真实IR收益与稳定性，动态同步节省的逐区域同范围前后量化仍需原版完整窗口补采。

主图保留原生起点和源码继承；同源码原IR复测单列灰色连接。固定交付历史、原始成绩和失败均保留。入口：[主图](../group-meeting-20261119/index.html#batchnorm)、[修改前决策](batchnorm/agent-calls/ir-004/before-edit-decision.json)、[实际IR审查](batchnorm/agent-calls/ir-004/ir-review.json)、[正式结果](batchnorm/evaluations/ir-004/result.json)、[稳定性](batchnorm/stability/ir-004-kgs-import-retry/summary.json)、[保留决定](batchnorm/selection.json)。
