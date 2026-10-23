# 会议剩余五算子达标批次

目标是五项正确性修复与性能达标。使用固定FlagGems/FlagTree、物理卡7、KGS19659完整调用walltime。各候选一次完整正式复验及补充语义、全部性能输入数值检查。服务器tmux连续运行，失败保留后转下一项；运行状态不等于最终交付。

对象：upsample_bilinear2d、_conj_physical、scatter_add、cumulative_trapezoid、fractional_max_pool3d。_conj_physical新建正确接口合同；scatter从上一批0.419×有效源码继续，其余从固定上游起步。合同和源码SHA见plan.json。

## 最终入口

本轮执行已结束，**没有五项全部达标**。scatter_add为1.900×；上采样0.340×，正确性通过但性能未达标；池化4.599×与私有共轭115.519×的参考含CPU回退，功能闭环、性能单列；累计积分参考存在数值不稳定问题。见[精简交付](../../会议剩余五算子交付-20261024.md)。

最终以`final-delivery.json`、`final-delivery-audit-v2.json`和`final-delivery-audit-v2.cases.csv`为准；四份源码集中在`deliverables/reviewed/<算子>/main.py`。原`endurance-state.json`、`deliverables/manifest.json`是主搜索结束快照，其中scatter旧1.306×有新增反例，池化尚未导入独立有效版；不能把这些旧字段当成最终清单。后续验证、取舍和源码身份记录在`finalization-state.json`、`packed-v2-state.json`、`packed-v3-state.json`和scatter修复批次。

## 复现与交接

新Linux副本先执行`git lfs pull`，再将原始证据归档展开到本批目录，恢复Windows短路径同步未展开的长路径原件：

```bash
B=project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2
tar -xzf "$B/raw-evidence/remaining-final-20261024-v1.tar.gz" -C "$B"
S=project/Ascend-Optimization-v1/operators/campaigns/scatter-segmented-20261023
tar -xzf "$S/raw-evidence/scatter-final-20261024-v1.tar.gz" -C "$S"
```

以上恢复文件不启动设备操作。原始记录使用910B绝对路径；在原910B仓库可直接离线重新核验，并指定全新输出避免覆盖：

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization
python3 project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2/knowledge/audit-final-delivery.py \
  --output /data/hanle/ascend-optimization/runs/remaining-five-20261023-v2/review-new.json
```

重新执行NPU测试时，先确认19655/19657/19659均空闲，并持有`/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock`；沿用`tle_yy`、Python3.11.15、CANN9.0、物理卡7及FlagGems/KGS的PYTHONPATH。独立门禁入口为`tools/profiling/hgkda/check-remaining-five.py`、`check-remaining-special.py`；scatter使用`check-endurance.py`和`check-scatter-dynamic-range.py`。传入最终源码及全新`--output`，不覆盖已有结果。正式评测仍用`tools/evaluation/evaluate-operator.py`，绑定本批`catalog/ops/<算子>`、`--timing-scope walltime --repeats 1`，使用全新输出目录；不得将CPU参考补验用时当作正式性能。

本轮四项已完整验收，不默认重启整个夜间搜索。若继续优化，使用新批次保留本轮合同和起点身份；优先处理上采样，累计积分先核对维护方的参考数值行为。

## 本轮踩坑

- 诊断导入oracle必须设置`sys.dont_write_bytecode=True`或`PYTHONDONTWRITEBYTECODE=1`。本轮探针曾生成catalog内`__pycache__`，导致累计积分和池化第二轮在模型启动前被KG严格目录校验拒绝。生成缓存已移至`research/generated-oracle-bytecode/`，合同未改；错误日志和迁移记录保留，不把这两次启动失败记为有效优化搜索。
- 主队列预算取消后，先核对worker的完成元数据和源码SHA；最后一次源码可能晚于先前快照。恢复日志的旧`error`字段不会自动清空，不能仅凭该字段判断服务仍停滞。
- 随机测试通过不排除算法错误。scatter的整行前缀相减版正式通过但动态范围0/18；最终显式分段版18/18，旧高分不交付。
- 低精度窗口运算的中间舍入会改变池化窗口及索引；数学上的整数倍率不等于可以忽略随机样本。普通cast在完整表达式中失败，而单独cast小探针正确，不据此指认某一条cast指令。
- 上采样连续块、编译期stride、打包存储均已保留失败或无收益证据。打包首版编译失败、宿主类型化指针版数值失败、显式编码版两case正确但变慢，未替换最终源码。
- `_conj_physical`与`conj_physical`不同。此次实数和complex64路径均核对了CPU回退、独立输出及共轭语义；不能引用旧公有接口成绩替代。

CPU回退参考与原生NPU参考分别标注，不通过调整容差或替换参考制造达标。最终源码在deliverables/，过程与失败见endurance-state.json、results/和research/。目标、方法预算与贡献见knowledge/夜间任务.md。

## 运行与验收

本批宿主tmux会话为`hanle-remaining-five-20261023-v2`，运行日志位于仓库同级`runs/remaining-five-20261023-v2/controller.log`。首轮每项最多4轮、50分钟；第二轮未达标对象最多3轮、40分钟。基线、补充门禁和正式评测另外计入总12小时预算。只使用物理卡7和串行卡锁，不重启共享服务。

本轮新增工具为`tools/profiling/hgkda/prepare-remaining-five.py`、`run-remaining-five.py`及`check-remaining-five.py`；初始SHA冻结于knowledge/tool-sha256.json。scatter/累计积分继续复用已核对的专用数值门禁。新三项除正式评测外，逐一核对所有correctness/timing输入，并补充布局、别名、特殊值和stream。

最终另外运行`check-remaining-special.py --operator <算子> --source <最终源码> --output results/<算子>/final-special.json`：上采样36项检查显式scale与布局，分数池化30项检查无穷、NaN、tie与采样stride。该追加工具SHA冻结于knowledge/supplemental-tool-sha256.json，不改初始合同或已冻结角色。原运行环境中执行时须沿用CANN、Python、PYTHONPATH和物理卡7配置。队列结束并下载完整原件后，用`audit-remaining-five.py --batch-dir <批次目录> --output <全新验收JSON>`离线核验。验收通过表示证据一致；是否超过0.8×、是否有CPU回退分别报告。

2026-10-23执行记录：本批62项正式正确性参考的CPU ABI预检全部通过；新增池化18项、上采样显式scale36项参考预检通过。上采样固定原版独立检查27/48，暴露1×1编译失败和stride寻址错误。首轮四次搜索均正式24/24，搜索最佳约0.345×，尚未达标；搜索成绩须通过独立门禁、正式复验及实际延迟选版后才能列为交付。实时状态以服务器endurance-state.json及进展.md为准，不把这段快照作为最终结论。

后续数值检查发现：scatter_add的8a67a004源码虽通过本批随机门禁，独立正式得1.306×，其全行前缀差仍存在跨target大数吞掉小结果的算法反例，因此该分数暂不作为最终有效交付。CPU反例见knowledge/scatter-range-cpu-counterexample.json；已排队后续scatter-segmented-20261023批次，在本批及补验退出后串行进行NPU反例核查和分段扫描修复。最终汇报必须结合后续动态范围验收，不能只根据本批原始门禁恢复这项达标声明。

累计积分已实测到参考自洽问题：同一FP16非末轴输入连续运行六次原PyTorch参考，380个位置的原容差允许区间交集为空；显式增量直接调用本机NPU cumsum同样超差。对照的FP16末轴、BF16及FP32未发现该容差自洽反例。原参考、合同和容差保持不变，该项仍不算完成完整数值闭环。版本、输入、逐项结果与复现探针见[实测说明](knowledge/累计积分参考自洽反例已实测.md)和`research/cumulative_trapezoid/reference-stability-probe/`。

后续scatter修复已有有效新结果：Codex显式分段扫描版正式18/18、补充18/18、性能输入数值9/9、新增动态范围18/18全部通过，完整调用相对PyTorch为**1.900×**，同case实际延迟较本批7cb有效起点整体改善**4.537×**。源码SHA为`72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e`，原件位于[修复批次](../scatter-segmented-20261023/README.md)。主队列的旧1.306×仍只是原门禁快照，最终汇报以新源码及动态范围证据为准；后续队列继续进行机制采集和有效版比较。

池化另有已验证新源码：`fractional_max_pool3d`的4228版正式9/9、完整输入与扩展27/27、特殊语义30/30均通过，完整调用为**4.599×，参考含CPU回退**。它在CC最后源码基础上由Codex修复低精度窗口位置的中间舍入；原始正式成绩及源SHA见[本批诊断与候选](Codex诊断与有效候选-20261023.md)。该结果已通过独立验收，仍需接入主队列最终选择、采集和同步；不能只读尚未收录该候选的旧状态字段判断结果不存在。
