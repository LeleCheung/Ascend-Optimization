# 严格 Triton 剩余五项攻关

本轮从提交`231ba49d69d6df45cfc51e6fd0fb04066b12e58d`的25/30继续，冻结已交付25项，不重测挑分。当前新增候选和原生参考仍在研究，交付数保持25/30。设备候选严格Triton；合格原生CANN或来源明确Ascend C只用于参考。保留原输入、容差、随机状态约束与KGS完整调用walltime。

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

`reference-probes/uniform-tensor-v1/`用于尚未测试的`aclnnInplaceUniformTensor`原生入口，固定原区间、三种类型及两种offset布局；不修改边界，不裁剪输出，不计性能。局部通过也不等于完整参考合格。

`analysis/sdk-reference-symbols-v1/`是单CPU只读清点：扫描1763份头文件、872份910B配置及7个opapi库的公开符号。已见`aclnnInverse`，但未在本次检查范围发现完整`inv_ex`或LU接口；GE头文件里的`FractionalMaxPool`是另一种语义，不能当作PyTorch的3D固定窗口接口。清单及上游fork源码只用于找线索，不作为参考资格证明，不据此断言所有厂商版本均不支持。

## 运行与恢复

910B项目路径：`/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1`。复用容器`tle_yy`和现有KGS；物理卡7、项目文件锁串行，不重启共享服务或镜像。

tmux会话为`hanle-strict-five-20261103`。三个阶段分别记录身份：

- `optimizer-process.json`：KG两轮搜索；每轮最多4稿、40分钟预算，原测量缓存复用。
- `half-evaluator-process.json`：等待搜索完成，再执行一次正式KGS、原63项及整块12项检查；同AST已有测量先复用。
- `reference-stage-process.json`：同一卡锁等待后构建及探测UniformTensor参考。

必须同时检查PID和`/proc/<pid>/stat`启动时间，不能只看名称或JSON声明。`state.json`的`tasks_finished`仅指先行采集队列完成；正在搜索的是`search-state.json`。未知请求不重发，在途工作区不修改；已完成任务不重复计时。

每份新候选仍需逐路径人工审查、运行时门禁、正式正确性、补测和实际编译／设备路径核验。PyTorch成绩单独保留；缺少合格原生参考时不计第26项。当前本批尚未完成，不能宣称30/30或三端最终交付。
