# 严格 Triton 剩余五项攻关

本轮从提交`231ba49d69d6df45cfc51e6fd0fb04066b12e58d`的25/30继续，冻结已交付25项，不重测挑分。**全部实验与选中稿核验已结束，严格CANN验收保持25/30。** `expm1_`新稿相对原PyTorch改善至0.646557×，通过全部选中稿门禁；新增原生UniformTensor探测未取得参考资格。结论见[交付报告](交付报告.md)、[30项总表](严格Triton30项总表.md)及[final-audit.json](final-audit.json)。设备候选严格Triton，合格原生CANN／来源明确Ascend C只用于参考；原输入、容差、随机状态约束与KGS完整调用walltime不变。以下保留起点与过程记录。

## 五项起点

|算子|候选已完成的验证|待解决问题|
|---|---|---|
|`cumulative_trapezoid`|正式33/33，补测34/36|FP16非末轴两项失败；原默认参考重复漂移已有容差区间无交集反例|
|`expm1_`|正式30/30，补测63/63；相对PyTorch 0.561436×|候选性能不足；原生参考的小值精度和负零检查未过|
|`uniform`|正式18/18，补测31/31；相对PyTorch 1.627599×|原生参考低精度输出超出原始请求区间|
|`linalg_inv_ex`|正式7/7，补测8/8|尚无包含`info`语义的合格原生参考；PyTorch计时含CPU回退|
|`fractional_max_pool3d`|正式9/9，补测15/15|尚无同语义原生3D入口；PyTorch计时含CPU回退|

源码、合同和完整旧交付冻结在`starting-points/`、`catalog/`、`starting-delivery.json`。上述为起点证据，不能当作本批新增测试。

## 本批新证据

`analysis/expm1_/paired-start-v1/`保存原FP16 `[4096,4096]`输入的真机成对profiling。候选和参考设备任务分别约130.263与28.221微秒；读写字节数相同，但MTE2指令数之比约8.561，MTE3约2.872，候选Vector冲突更高。流水时间存在重叠，不能相加；单项活跃比例也不能证明HBM受限。结构化比较及旧失败轨迹已注入KG新搜索上下文。

`proposals/expm1_/codex-native-half-v1/`是Codex提出的FP16数学提案：保留FP16指数计算，小幅值使用二阶公式；其他类型、布局和启动继承冻结最佳。Triton数学内置经正常编译链生成`math.exp`，不调用宿主CANN。六份离线编译通过及65536种FP16位模式的CPU舍入模型，只证明编译和理想公式；真机精度与性能须独立评测。可复现模型见`tools/profiling/hgkda/strict_five/model_half_formula.py`；`native-half-formula-domain-v2.json`补齐脚本SHA和中文范围说明，旧v1描述乱码原件保留。

实际TTIR进一步发现v1的两个比较常量仍使FP16值扩展到FP32。`codex-half-comparisons-v2/`只将阈值和零常量显式设为FP16；六份离线编译通过，两种FP16 kernel 的TTIR均不再包含`arith.extf`。该变化只证明编译表达改变，未证明真机更快或数值通过；独立评测器等待v1终态，再持同一卡锁验证。

KG第二轮第四稿`8ec3ef796d4bdbe7…`已完成一次正式KGS：30/30、相对PyTorch 0.646557×，比冻结0.561436×提高约15.16%。它按dtype调整数学表达、阈值和流式分块。人工逐函数审查已保存于`reviews/expm1_/`；现已通过63/63补测、12/12整块、4/4完整FP16域路径及实际编译／设备任务核验，作为原PyTorch口径新最佳保留。性能与CANN参考仍有缺口，不计新增CANN达标。源码注释中的局部耗时与瓶颈判断不作为已核实机制结论。

`reference-probes/uniform-tensor-v1/`已实际探测`aclnnInplaceUniformTensor`原生入口，固定原区间、三种类型及两种offset布局。6/12通过，其余低精度结果低于请求下界。不修改边界、不裁剪输出、不计性能，未取得完整参考资格。

`analysis/sdk-reference-symbols-v1/`是单CPU只读清点：扫描1763份头文件、872份910B配置及7个opapi库的公开符号。已见`aclnnInverse`，但未在本次检查范围发现完整`inv_ex`或LU接口；GE头文件里的`FractionalMaxPool`是另一种语义，不能当作PyTorch的3D固定窗口接口。清单及上游fork源码只用于找线索，不作为参考资格证明，不据此断言所有厂商版本均不支持。

继续检查旧图接口后，`legacy-kernel-symbols-v2/`补充发现GE的`Lu`及设备AiCPU配置：输出为分解矩阵和置换，仍不含`inv_ex`的`info`；这与“ACLNN未发现LU接口”是不同范围。该路径尚未执行，不能计为合格参考。v1漏掉大小写及嵌套符号目录，v2原始匹配混有`%lu`格式串；使用保留完整原件并排除格式串的`filtered-summary.json`查看线索，不能从原始命中数推导算子支持数量。

`analysis/installed-expm1-impl-v1/`保存本机TBE及Ascend C源码线索。旧TBE存在按设备能力选择`exp(x)-1`或更稳定公式的分支；安装的v35源码属于另一架构路径，`HIGH_PRECISION`宏或源码文件存在都不能证明910B当前二进制已执行稳定公式。当前参考资格仍以实际小值和负零失败为依据。

## 运行与恢复

910B项目路径：`/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1`。复用容器`tle_yy`和现有KGS；物理卡7、项目文件锁串行，不重启共享服务或镜像。

tmux会话为`hanle-strict-five-20261103`。六个阶段分别记录身份：

- `optimizer-process.json`：KG两轮搜索；每轮最多4稿、40分钟预算，原测量缓存复用。
- `half-evaluator-process.json`：等待搜索完成，再执行一次正式KGS、原63项及整块12项检查；同AST已有测量先复用。
- `half-comparisons-process.json`：等待首个提案终态，再独立验证去FP32比较扩展版本。
- `reference-stage-process.json`：同一卡锁等待后构建及探测UniformTensor参考。
- `half-domain-process.json`：等待两份FP16提案与参考探测终态，给通过先行检查的提案补测全部FP16位模式的四类真实路径，不计性能。
- `reviewed-best-process.json`：搜索终态后复用0.646557×源码的已有正式成绩，检查原补测、整块特殊值和FP16完整域，再核验实际Triton编译／设备路径。

必须同时检查PID和`/proc/<pid>/stat`启动时间，不能只看名称或JSON声明。`state.json`的`tasks_finished`仅指先行采集队列完成；正在搜索的是`search-state.json`。未知请求不重发，在途工作区不修改；已完成任务不重复计时。

选中稿已通过逐路径人工审查、运行时门禁、正式正确性、补测及实际编译／设备路径核验。PyTorch成绩单列；缺少合格原生参考不计第26项。7个阶段及2个KG工作区均已终态，不得重启旧入口。离线复现使用`python tools/profiling/hgkda/strict_five/finish.py`，不提交设备测试；同步状态另见`三端同步核验.json`及实际Git提交，不从实验终态推定同步完成。
