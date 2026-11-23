# 上采样：固定源码，删除无效lane填充值

当前状态：完整合同24/24、双方各30/30补测、ABBA、修改后profiling及模拟器机制审查通过，已保留新IR。正式／重放二进制一致，30/30个PC时钟核对通过；超时退出、部分窗口和PAD预测偏差保留。

固定输入为FP16 `[4,32,128,128]→[4,32,256,256]`，`upsample_bilinear2d-timing-1`。Triton源码SHA为`a0e1e0d687176b904af29fbac46b505a7ccd8fe76fc0ea6add1ccad350ab1110`，从前夜已保留的FP16边界选择IR继续。父IR `129be797…`，候选IR `8d18503a…`，编译器原IR身份另存于清单。

修改前模拟器部分窗口中，MTE2热点包含逐行搬运及重复设置PAD值，Vector同时有有效工作与等待。Codex提出关联行复制与向量工作成本，模型实际读取profiling、热点、流水、TTIR及BCMLIR后，选择删除9处masked load的`other=0`，以及相应死常量。原指针、掩码、FP32插值算术、六个FP16边界选择、打包和写回保持固定。

有效写回lane消费的邻居均有效，边界邻居由原select回退到有效中心；其他lane受原store掩码屏蔽，kernel内无跨lane归约。宿主审查覆盖90组边界形状、290088个有效输出lane。此审查验证本次增量变换的消费条件，父版既有整行copy行为和验证范围继续单独保存。

实际CPU BCMLIR中FP16零填充3→1、条件区域2→0、insert_slice7→3；9处copy及6个行循环保留。正式全合同25个AST中，仅本次登记目标二进制变化，其余24个逐字节一致。

| 测量 | 保留IR | 新IR |
| --- | --- | --- |
| 本阶段完整合同目标耗时 | 159.228890μs | 146.503486μs |
| 同轮配对目标／CANN | 1.537515× | 1.663731× |
| ABBA区块中位数 | 156.742455μs | 145.619453μs |
| 独立profiling每程序Vector时间 | 3.005380μs | 2.633945μs |
| 独立profiling每程序Scalar等待 | 3.050487μs | 2.611525μs |
| 独立profiling每程序MTE2指令计数 | 201 | 201 |

正式固定shape耗时下降7.9919%。全部性能case候选耗时的几何比较为原／改1.010894×。两次原始配对geomean为1.067412／1.061804；采用新候选逐case CANN共同锚点后为1.050361／1.061804。参考变化和未修改二进制的测量变化继续保留，主图按逐case核验后重算。

模型预期PAD指令减少及MTE2活动变化，实际profiling的MTE2指令计数保持201，向量工作与标量等待下降。新重放仍有重复MOV_PAD_VAL编程，原预测未兑现；实际低层fill／条件／切片减少与Vector时间下降支持可确认的机制。前后窗口边界分别为145.032～153.026及134.477～145.028原始时钟位置，各自归一化；动态执行覆盖量不同，原始指令总数和泳道比例分别解释。修改后30/30个PC时钟核对通过，应用因超时以状态9退出，直接PC源码映射范围继续保留。

本轮固定源码用于隔离局部编译变换。同类意图可在Triton源码表达，源码表达与自动pass迁移属于后续验证。候选由CC／DeepSeek实现，Codex的诊断、等价检查、基础设施和验收贡献分别保存。

材料：[修改前决策](upsample/agent-calls/ir-001/before-edit-decision.json)、[IR diff](upsample/agent-calls/ir-001/actual-ir.diff)、[宿主审查](upsample/agent-calls/ir-001/ir-review.json)、[低层机制](upsample/agent-calls/ir-001/lowering-mechanism.json)、[完整结果](upsample/evaluations/ir-001/result.json)、[二进制核对](upsample/ir001-binary-comparison.json)、[ABBA](upsample/stability/ir-001/summary.json)、[后profiling](upsample/post-mechanism-ir001-profiling.json)。
