# 当前最佳版本的IR性能研究

IR阶段延续持续模拟器重放闭环：重放当前版本并定位热点与等待 → 选择Triton／IR修改层 → 编译修改版并再次重放 → 结合完整真机验收保留最佳。既有案例先完成真机评测，再采集修改后profiling及模拟器重放；执行顺序和窗口范围按原日志保留。详见[重放表述纠正](../presentation-replay-correction-20261120.md)。

当前已有3项完整验收IR收益，进度3/3。三项均冻结Triton源码，通过完整KGS、必要补测、ABBA及修改后profiling／模拟器双分析。完整记录见[夜间IR记录](../night-ir-campaign-20261119.md)。

scatter_add新增案例为二分尾步折叠：generic gather从14降为13，其他26个AST二进制逐一相同；原／改版两段ABBA均确认方向。修改后模拟器为部分窗口，79/79个逐PC时钟核对通过，与修改前152/152及各自窗口分别保存。QKV `ir-007`／`ir-008`完整及补测通过，63.038／63.666μs均未超过62.861μs原IR；[388份](qkv-ir007-no-gain-receipt.json)及[437份无收益记录](qkv-ir008-no-gain-receipt.json)均已导出并逐文件校验。

| 算子 | 原IR→修改IRμs | 相对CANN | 完整与补测 | 材料 |
| --- | --- | --- | --- | --- |
| BatchNorm | 42.955284→39.261332，下降8.60% | 0.717359× | 27/27；双方各36/36；ABBA与双采集通过 | [案例](batchnorm-ir004-case.md)、[选择](batchnorm/selection.json)、[402份收据](batchnorm-accepted-ir004-complete-inputs-receipt.json) |
| 上采样 | 164.856213→159.569697，下降3.21% | 1.538096× | 24/24；双方各30/30；ABBA与双采集通过 | [案例](upsample-ir003-case.md)、[选择](upsample/selection.json)、[285份收据](upsample-accepted-ir003-receipt.json) |
| scatter_add | 1281.927461→1262.722303，下降1.50% | 2.721707× | 18/18；双方各21/21；专项3/3；ABBA与双采集通过 | [案例](scatter-ir002-case.md)、[选择](scatter/selection.json)、[474份收据](scatter-accepted-ir002-receipt.json) |

三项新节点已接入主图、固定输入曲线和[22页持续重放修正版PPT](../group-meeting-20261119/whitebox-simulator-ascend-20261120-replay.pptx)。11月20日更新为[两组八图](../presentation-eight-figures-20261120.md)：四张全用例geomean与四张固定shape机制，QKV继续保留源码最佳。26图与四张固定输入图的连续／单调检查通过，更新截图已检查。以下材料说明同时保留此前各轮过程。

BatchNorm固定输入为FP32 NCHW `[8,64,64,64]`、`timing-7`。冻结Triton源码为`b8d4d4a2c07d…`，本轮源编译基线完整27/27、45.318202μs、0.620374×CANN。

CC／DeepSeek将EVEN路径Welford的均匀计数从`8×128`整数循环状态改为运行时循环步数乘积与标量转换，保留原浮点递推、访存和函数接口。原IR与修改IR均通过当前原生解析和模块校验；修改版完整27/27、双方各36/36补测通过，目标45.173407μs、0.619499×CANN。原IR完整复测42.955284μs、0.659340×CANN，因此本轮保留原IR，收益计数保持0。

源编译基线与原IR覆盖产生相同二进制，两次耗时相差约5.5%；该差异作为测量波动保存。修改版产生新的二进制，profiling与模拟器复采均已完成；模拟器保留部分窗口、逐PC时钟核验及相对流水占比。第二轮已恢复原IR字节，由模型选择更大的循环／搬运机制继续。

- [修改前决策](batchnorm/agent-calls/ir-001/before-edit-decision.json)
- [模型机制说明](batchnorm/agent-calls/ir-001/output-diagnosis.json)
- [宿主语义审查](batchnorm/agent-calls/ir-001/ir-review.json)
- [原生解析检查](batchnorm/agent-calls/ir-001/parse-review.json)
- [完整源编译基线](batchnorm/evaluations/frozen-baseline/result.json)
- [导出收据与远端原始trace身份](reviewed-batchnorm-ir001-receipt.json)

首份原IR覆盖的目标case遇到输入／序列化文本SHA差异断言，失败原件保留在`batchnorm/evaluation-history/`。通用加载器已分别绑定原始IR输入、原生解析输出与缓存stage；修复后原IR与修改版均完成正式评测。

QKV第二份IR候选已完成原生解析及语义审查。它展开固定两轮块循环，并提前两块的六次加载；Q路径保留FP32中间计算，k/v保留原FP16加法。源编译基线完整21/21、62.957504μs、1.449662×CANN。

CPU编译核对中，原版BCMLIR及二进制与已执行基线SHA完全一致；修改版BCMLIR的静态循环数从1变为0，二进制SHA发生变化。两版均编译到二进制，NPU保持未初始化。静态拷贝节点按循环展开出现不同计数，实际读写量、流水重叠与性能继续由设备复采验证。

- [QKV修改前决策](qkv/agent-calls/ir-001/before-edit-decision.json)
- [QKV语义审查与证据范围](qkv/agent-calls/ir-001/ir-review.json)
- [QKV低层编译记录](qkv/agent-calls/ir-001/lowering-review.json)
- [QKV产物变化](qkv/agent-calls/ir-001/lowering-mechanism.json)
- [QKV导出收据](reviewed-qkv-ir001-receipt.json)

QKV `ir-001`正式21/21、双方各48/48通过，目标63.029176μs；原IR为62.860936μs，继续保留原版。相对CANN比值的轻微增加来自参考耗时变化。第二轮继续调查连续加载宽度与输出布局，首轮修改后双采集仍在设备锁后排队。

BatchNorm `ir-004`与上采样`ir-002`已完成宿主审查及CPU编译。前者仅加宽归一化／写回至8×256，统计路径保持原字节；后者修复邻行加载边界并在实际BCMLIR中消去六个逐行复制循环。两项设备收益均待完整正式评测。

- [BatchNorm归一化加宽决策](batchnorm/agent-calls/ir-004/before-edit-decision.json)
- [BatchNorm语义审查](batchnorm/agent-calls/ir-004/ir-review.json)
- [BatchNorm原／改IR编译记录](batchnorm/agent-calls/ir-004/lowering-review.json)
- [上采样边界修复决策](upsample/agent-calls/ir-002/before-edit-decision.json)
- [上采样语义审查](upsample/agent-calls/ir-002/ir-review.json)
- [上采样低层机制变化](upsample/agent-calls/ir-002/lowering-mechanism.json)
- [31份审查及编译材料的导出收据](reviewed-bn004-up002-receipt.json)
- [BatchNorm首轮正式结果及修改后摘要导出收据](batchnorm-result-ir001-receipt.json)
