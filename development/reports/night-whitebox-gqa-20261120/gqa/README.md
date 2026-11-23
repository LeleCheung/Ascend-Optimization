# GQA：减少KV循环交接的完整四项案例

目标case4：**1378.73655 → 784.63765 μs，耗时下降43.09%**。
BF16非因果，Q `[4,256,64,128]`、K/V `[4,256,8,128]`、G8。新版本相对本轮CANN为0.160644×。
固定官方case1/2/4/7全部通过；完整20项验收仍待后续开展。

| case | 本轮原保留源码 μs | 新源码 μs | 耗时下降 | 本轮CANN μs | 相对CANN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 180.91910 | 179.09665 | 1.01% | 91.02455 | 0.508243× |
| 2 | 183.68115 | 181.33365 | 1.28% | 87.23110 | 0.481053× |
| 4 | 1378.74995 | 784.63765 | 43.09% | 126.04765 | 0.160644× |
| 7 | 194.95115 | 181.49265 | 6.90% | 107.62710 | 0.593011× |

四项候选耗时几何平均下降15.17%，几何加速1.178872倍。
两边合同、harness、官方checker完全相同：warmup5、trials3、iterations20。目标三次原版为
[1378.80195, 1378.74995, 1378.33195]μs，新版为[785.0641499999999, 784.39465, 784.63765]μs；本轮原版1378.74995μs与历史成绩接近。
当前差距明确，按条件策略省略新增ABBA。

修改前模拟器的热点为WAIT_FLAG_DEVI与FIX_L0C_TO_DST，Cube窗口productive4.74%、FLOWCTRL等待89.79%；
Vector0窗口productive35.98%、等待29.87%。结合实际TTIR与BCMLIR，模型选择减少KV块循环的Cube／Vector交接。
`BLOCK_M=64`保持，D≤128的非因果输入将`BLOCK_N=64→128`，case4的KV轮数4→2。
实际改动为宿主分块选择，kernel函数AST及两项概率分解表达式保持；在线softmax归约分组变化由官方checker验证。

首版全部D≤128都用BN128，case1编译需要231488B UB、可用196608B，超出34.0625KiB。
修复版按真实`is_causal`恢复因果BN64、保留非因果BN128。模型对资源峰值的其他估计作为假设保留。
两次调用均成功读取相应证据，并在编辑之前登记决策；[首版审查](agent-calls/attempt-001/host-review.json)、
[修复审查](agent-calls/attempt-002/host-review.json)、diff、完整反馈与真实父链均已保存。

修改后profiling：kernel1368.927→791.996μs；每program MTE1指令45→23、MTE2指令18→10，
Vector0服务9.344→6.780μs、Scalar等待5.809→4.366μs，L0C→GM320→192KiB，GM→UB320→192KiB。
该数据支持减少交接与中间数据搬运。硬件服务时间、单kernel耗时和正式完整调用耗时分别记录。

前后三核重放时钟均核验通过：原版19/19、41/41、38/38，新版19/19、43/43、40/40个PC。
新版Cube窗口productive5.19%、FLOWCTRL等待83.51%；Vector0窗口productive29.17%、等待22.87%。
各窗口分别归一化，VECTOR的可观测空闲从34.15%增至47.96%，其他等待与占比变化也同时保留。
两次应用均未完整结束；部分窗口比例用于解释局部执行，完整收益采用真机四项结果。
每个版本的正式评测cache、硬件profiling与模拟器cache中目标npubin字节一致，见
[修改前绑定](binary-binding-and-facts.json)、[修改后绑定](iterations/attempt-002/binary-binding-and-facts.json)。
GQA实际修改位于Triton宿主侧，IR用于机制分析；后重放持续执行。

[连续最佳曲线](../../gqa-simulator-20261120/index.html) · [保留决定](selection.json) ·
[完整四项结果](iterations/attempt-002/parent-validation/results.json) · [完整源码／决策trace](trace/rounds/attempt-002/round.json)

原始大trace继续保存在远端`/data/hanle/ascend-optimization/runs/night-whitebox-20261120/gqa/`及
`iterations/attempt-002/simulator-parent/`。本地保存源码、实际IR、目标二进制、完整结果、profile表与重放摘要。
原始模型结果回读及[宿主纠正](../model-result-review-host-note.md)共同保留。
