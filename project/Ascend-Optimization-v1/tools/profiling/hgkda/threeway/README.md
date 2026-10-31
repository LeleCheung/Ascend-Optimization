# 三路对照与剩余算子优化

2026-10-30启动，批次为`operators/campaigns/threeway-20261030/`。沿用固定FlagGems、FlagTree、CANN和原KGS合同，使用910B物理卡7及本项目卡锁。新工具与构建产物均放独立目录。

## 目标与顺序

1. 原样复验12项已达标源码和`log10_`，并核对`cross_out`、`nan_to_num_`、`native_group_norm`起点，共16项。
2. 先接入`log10_`、`hardswish_out`、`native_group_norm`的CANN原生库参考，沿用相同输入、原地/out、数值容差和计时参数。
3. 检查自写Ascend C `log10_`候选：连续且对齐输入采用Ascend C，其他布局沿用已验证Triton路径。明确标注混合实现，不写成纯Triton或厂商最佳算子。
4. 在相同代表输入上采集设备任务路径。独立profiling耗时用于解释执行差异，不代替正式KGS加速比。
5. 验证分块、分派和宿主开销改进，再对未达标的重点算子执行KG/CC/DeepSeek搜索。全程保留失败方案及完整检查结果。

## CANN计时与资源

正式C++参考由`cann_cpp.cpp`和`build_cann_cpp.py`构建。它直接调用公开aclnn API，使用torch_npu提供的张量分配和任务队列，不调用aten目标算子。`log10_`、`hardswish_out`和归一化的第三版包装均已完成完整合同和补测。归一化第三版先提交同stream的宿主队列，随后在同线程query/execute，不额外等待设备；此前失败包装仍保留。CANN库内部源码未核验，因此称“CANN原生库参考”，不称为已核验Ascend C实现。

`cann_async.py`不在正常调用内全局同步。它先刷新PyTorch宿主任务队列，使前序操作提交到正确设备stream，再直接调用aclnn。描述符、工作区及张量由stream事件保留到设备执行完成，之后回收。不能仅删除旧包装中的同步而立即释放这些资源。

完整调用仍包含本Python ctypes包装、事件管理和分派成本，因此该成绩不代表最佳C++宿主封装。PyTorch也可能使用同一个aclnn实现，需要结合实际设备kernel判断三路是否只是调用层不同。参考包装使用KGS原生`python`入口类型，不伪装成Triton候选；代码与正式请求均留档。

## 文件

|文件|作用|
|---|---|
|prepare.py|复制并核验原合同与原版源码，单独冻结最新有效起点|
|run.py、launch.py|项目卡锁下串行复验、参考对照、候选验证与KG搜索；tmux不依赖控制端持续在线|
|references.py、evaluate_reference.py|同KGS合同评测CANN与源码可核验Ascend C；参考结果不进入Triton候选择优|
|cann_async.py、test_lifetimes.py|异步资源生命周期与无需NPU的检查|
|paths.py|同输入实际设备路径，保留原trace和区间统计|
|ascendc/、build_log10.py、make_native.py|自写Ascend C候选、单CPU构建及源码/二进制SHA冻结|
|make_proposals.py|按历史证据生成分块与宿主分派候选，不预先认定性能提升|
|status.py|只读报告本批状态，不读取模型思考记录|
|transfer.py|通过SSH标准输入上传与回传，逐文件核验SHA|
|finalize.py、report.py|对新过线源码作同条件完整确认，核验请求、合同、逐case汇总与补测；两份有效结果在简表汇总为一个几何平均值|
|reproduce.py|宿主单命令复测当前交付源码，自动取得卡锁，结果另存，不替换历史最佳|
|trace_summary.py、reparse_paths.py|按实际Ascend Hardware／Task Type解析设备任务，从原trace重算摘要，无需重复使用NPU|
|search_context.py|将本轮完整成绩和已否定改法提供给后续KG搜索，明确区别于新采集的metrics|

## 运行与接续

从仓库项目目录运行。批次目录已存在时不要再次运行`prepare.py`，也不要删除未知请求以强行恢复。

```bash
python tools/profiling/hgkda/threeway/prepare.py --batch threeway-20261030
python tools/profiling/hgkda/threeway/make_proposals.py
```

910B宿主启动：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/threeway/launch.py --phase all
```

各阶段只创建本项目对应的tmux会话，并使用同一卡锁。启动多个等待者时，Linux锁不保证按启动顺序交接；有明确先后依赖的阶段应在前一阶段退出后启动。已知完整请求不会重复提交。日志在`/data/hanle/ascend-optimization/runs/threeway-20261030/`；进展在批次`next-state.json`、`进展.md`中。

Ascend C构建使用独立的`runtime/threeway-ascendc-log10-build/`，固定`SOC_VERSION=Ascend910B4`、`CMAKE_BUILD_TYPE=Release`、`-j1`。空的构建类型会导致CANN合并目标脚本失败；首个失败清单与后续成功清单均保留。设备语义验证通过前，不把编译成功称为算子完成。

通过SSH读取状态：

```bash
ssh baai-910b python3 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/threeway/status.py
```

本次Windows OpenSSH `scp`曾返回成功但远端没有生成文件。`transfer.py`改用SSH标准输入及SHA回执，避免以退出码替代实际文件核验。最终收尾需回传批次并再次核验，不能将运行中的快照称为最终同步。

## 单命令重新评测交付源码

在910B宿主的项目目录执行，例如：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/threeway/reproduce.py --operator log10_ --run-name manual-20261031
```

将`log10_`改为`native_group_norm`可复测归一化；重复实验使用新的`--run-name`。工具按实际manifest选择Triton或Python/Ascend C入口，在原KGS合同下做一次完整评测和独立补测，结果写入`campaigns/threeway-20261030/reproductions/`。它不会修改历史最佳、镜像或服务。

当前原生二进制固定于aarch64、Torch 2.9.0、CXX11 ABI与CANN 9.0.0，加载时校验SHA。迁移服务器需要按构建记录重建并登记新候选，不能直接改掉冻结源码中的路径或SHA。

## 本轮已修复及保留的问题

- `compact-contiguous-host-v2`归一化候选改名后遗留早期别名，导入失败；`v3`修复该入口，旧失败保留。
- 起初只冻结了16个算子叶子目录，缺少KG需要的根`catalog/manifest.json`。补齐目录元数据后使用新的`attempt-2`，四份正式合同文件的原字节与SHA保持相同。首个失败发生在`kg run`之前，没有工作区或已提交设备请求，见`catalog-preflight-resolution.json`。
- Ascend Chrome trace的硬件任务可以没有`cat=kernel`；原摘要因此漏报。新解析按`Ascend Hardware`进程和`Task Type`识别，并排除Overlap Analysis重复轨道，使用Decimal处理绝对时间。原trace和旧摘要保留，正确重算结果写入`summary-v2.json`与`path-index-v2.json`。缺记录不能当成设备零耗时。
- CANN归一化C++任务队列包装在连续计时中出现资源问题。失败版本独立保留；第三版将query/execute放在同一宿主线程，通过原合同及补测后替换展示参考。这个修正属于参考包装，不计为候选算法优化。
- `cross_out`的CC高分源码一度只按地址、dtype和元素数命中分派，缺少dim/stride复核。修复版移除此memo，恢复真实storage别名保护并补齐移位读取上界。新增`check_cross.py`覆盖同地址改维度、部分尾块、共享非连续storage、特殊值和当前stream；旧高分源码退出成功选择与知识索引，原始结果保留。

所有正式成绩均为完整调用walltime。独立profiling中的设备区间、同进程补测的宿主提交及含同步完成时间分别记录，不替换KGS验收成绩。
