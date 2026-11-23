# BatchNorm：均匀整数计数标量化

完整27/27、双方各36/36补测通过，固定shape为40.024682μs，慢于本阶段保留版39.458217μs和历史最佳39.261332μs。本轮保持夜间8×256输出IR，后profiling、模拟器重放及正式二进制身份审查已完成。稳定性按预先设置的目标收益筛选跳过，384份完整归档证据逐文件SHA校验通过。

输入为FP32 NCHW `[8,64,64,64]`、`timing-7`。Triton源码固定为`b8d4d4a2c07dabfc308e94412d0b5c219eb56ee4c7dc5358016e0eb2782b2f56`。上一调用的广播外提最终机器码相同，宿主逐字节恢复保留IR后再开展本次实验。

修改前保留二进制的模拟器探针记录：S32 VADDS在同一计数缓冲更新64次，相应S32→FP32 VCONV读取2次，同核可见2次END。统计循环每程序32轮，与该重复计数对应。该关联来自指令数据流与实际IR结构，直接PC源码映射范围继续保留。

模型将统计循环携带的`tensor<8x128xi32>`改为标量i32；每轮仍加一，循环后标量转换为FP32，再广播供原浮点计算消费。所有原计数lane从零开始、每轮无条件加一，通过归纳可知每次迭代后均与标量相同，含零迭代和i32模运算。宿主逐行核对其余IR完全相同，访存、循环边界、输出分支和浮点顺序保持一致。

实际CPU编译通过：BCMLIR张量整数计数加法及转换消失，标量循环携带、标量转换和最终fill保留；静态循环6处，fill16→15。原／改二进制为8416／8264字节，SHA分别为`41483d74…`／`28106c7d…`。完整评测28个AST中仅登记目标1个二进制变化，其余27个逐字节一致。

| 指标 | 保留IR本阶段复测 | 新IR |
| --- | --- | --- |
| 固定shape耗时 | 39.458217μs | 40.024682μs |
| 同轮目标／CANN | 0.709213× | 0.702994× |
| 原始配对全合同geomean | 1.832020× | 1.868342× |

全合同候选耗时原／改几何比较为1.023075×。目标特化变慢与其余同二进制特化的跨事件测量变化分别保存。本次目标未超过保留版，主图继续采用已验收版本。

修改后重放与正式二进制相同，30/30个非Scalar、非同步PC时钟核对通过；前后同核均有2次END。S32向量VADDS64→0、S32→FP32向量VCONV2→0，均兑现原预测；标量ADD316→318、标量S32F32 CONV68→70。后端进一步优化了标量循环携带，因此实际新增标量工作按动态指令计数记录。

独立profiling双方均64个程序、1650MHz：每程序Vector9.515303→8.882737μs、Scalar4.645445→4.415938μs、AIV19.289365→18.976809μs；同时Scalar等待1.502008→1.651477μs、MTE3时间2.902784→3.074242μs。独立profiling任务39.880798→38.780777μs与完整合同目标40.024682μs分别保存；两类测量保持各自合同，性能变化的独立关键路径原因仍待研究。

前后窗口分别归一化，直接PC源码映射范围保留。新应用产物清单已生成，包装进程返回0、子进程退出139继续原样保存。当前机制结论为向量计数工作确实移除，目标真机收益验收未通过。

读取顺序核对确认：热点、流水、成对profiling、源码、TTIR及计数指令探针均在decision成功前读取，随后由模型完成7处编辑。本轮日志记录了低层清单读取，实际BCMLIR内容由Codex在交接后审查；模型最终文字中更广的读取声明保留为原始记录，宿主记录按实际日志区分时序。固定源码用于隔离均匀性传播这一编译变换，同类标量计数意图可在Triton表达。

材料：[修改前决策](batchnorm/agent-calls/ir-005/before-edit-decision.json)、[读取顺序核对](batchnorm/agent-calls/ir-005/evidence-read-review.json)、[实际IR diff](batchnorm/agent-calls/ir-005/actual-ir.diff)、[语义审查](batchnorm/agent-calls/ir-005/ir-review.json)、[CPU机制](batchnorm/agent-calls/ir-005/lowering-mechanism.json)、[完整结果](batchnorm/evaluations/ir-005/result.json)、[双补测](batchnorm/evaluations/ir-005/supplementary-summary.json)、[二进制比较](batchnorm/ir005-binary-comparison.json)、[保留决定](batchnorm/retention-ir-005.json)、[后重放机制核对](batchnorm/post-mechanism-ir005.json)、[正式二进制绑定](batchnorm/observation-binding-ir-005.json)、[完整归档清单](exports/batchnorm-ir005-negative-complete.tar.gz.receipt.json)。此前53份CPU阶段证据另存。
