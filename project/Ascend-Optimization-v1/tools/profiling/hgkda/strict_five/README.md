# 严格 Triton 剩余五项闭环

本批实验已结束，验收保持25/30；`expm1_`原PyTorch口径提升至0.647×并完成全部选中稿门禁，其余数值和原生参考缺口单列。见[交付报告](../../../operators/campaigns/strict-five-20261103/交付报告.md)。先读终态，不重新执行已有启动脚本。

在`project/Ascend-Optimization-v1`目录运行`python tools/profiling/hgkda/strict_five/finish.py`，可在Windows／Linux离线核验冻结合同、完整结果、补测、编译产物和终态身份，并重建30项总表及CSV；不使用NPU、不重新计时。要求Python3.11及以上；910B使用`/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python`运行该脚本，宿主默认Python3.7不适用。`final-audit.json`证明证据与门禁核验，不表示30项均达标。

以提交`231ba49d6`的严格Triton交付25/30为起点，冻结已有25项，仅继续研究`cumulative_trapezoid`、`uniform`、`fractional_max_pool3d`、`expm1_`、`linalg_inv_ex`。

参考使用合格原生CANN／来源明确的Ascend C；候选全部设备计算、搬运和回退严格Triton。原PyTorch正确性合同、容差、输入和KGS完整调用walltime保持不变。原生参考不合格或缺失单列，不通过较慢自写参考、放宽容差或改变确定性设置凑30项。

`prepare.py`冻结上一轮交付和五项原合同，建立独立批次`operators/campaigns/strict-five-20261103/`。设备操作均持物理卡7项目锁串行，复用现有容器和服务；不重启共享服务，不修改镜像，不触碰其他用户进程。

先检查原生入口及参考适配的新线索，再围绕`expm1_`的大输入设备差距采集真实流水指标，结合源码、IR及必要的小实验修改候选。累计积分继承默认参考漂移的固定反例；同一失败原因没有新证据时不重复搜索。候选经门禁、完整KGS、补测、人工审查及实际后端核验后，才进入交付。

`optimize.py`负责串行KG搜索；`evaluate_half.py`验证保留FP16数学运算的提案，`evaluate_half_comparisons.py`等待前者终态后验证进一步消除比较扩展的版本。`reference_stage.py`构建及探测原生UniformTensor参考。每个阶段有独立进程身份，不得重复执行启动命令；实时状态和恢复注意事项见批次README。

`inventory_references.py`只读扫描安装头文件、910B配置及opapi符号；`model_half_formula.py`只做CPU舍入模型。两者均不使用NPU，不产生正式性能成绩，也不能代替原生参考资格或候选真机验证。

`probe_half_domain.py`在实际平铺、大流式、尾部和非对齐路径验证全部FP16位模式；`half_domain_stage.py`将其排在在途提案之后。`verify_reviewed_best.py`绑定已经逐函数审查的KG第二轮新稿，复用已有完整KGS成绩，先补测再核验后端，不重复正式计时。最终验收还须核对这些新检查，不能仅凭早先的63项补测或离线编译交付。

`paired.py`是本批实际使用的成对采集器：reference显式标为冻结原PyTorch，而非合格CANN。它的字节身份同时记录在采集任务与结果中，随仓库保留，不能漏掉仅在服务器生成的工具。
