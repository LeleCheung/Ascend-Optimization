# 严格 Triton 夜间优化

以 `strict-triton-20261102` 已冻结的 **16/30** 为起点，参考为合格的原生 CANN／来源明确的 Ascend C，候选所有设备计算与搬运均使用 Triton。禁止通过候选调用 CANN、Ascend C、宿主原生 DMA、PyTorch 或 CPU 数值计算替代。

优先顺序：`hardswish`、`fix`、`hardshrink_backward`、`rad2deg_`、`round_out`、`special_round_out`、`hardswish_out`、`log2_`、`fmod_tensor_`。随后继续处理 `cumulative_trapezoid`、`linalg_inv_ex` 的数值问题，以及 `expm1_`、`uniform`、`fractional_max_pool3d` 的参考缺口。25/30 是优化目标，以实际结果为准。

## 运行

910B 宿主入口：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python -u tools/profiling/hgkda/strict_night/run.py
```

批次为 `operators/campaigns/strict-night-20261102/`，服务器会话为 `hanle-strict-night-20261102`，控制日志为 `/data/hanle/ascend-optimization/runs/strict-night-20261102/controller.log`。

队列在项目物理卡7锁内运行，复用 `tle_yy` 与现有 KGS，不重启服务，不修改镜像。KG／HG-KDA 使用独立源码覆盖层，CC／DeepSeek 实际生成候选，Codex 提供诊断、编排与审查。每项最多两轮搜索，每轮最多4个正式优化轮次，单次搜索预算40分钟，运行中的正式评测有有界收尾时间。

## 验收

1. 原始输入、容差和计时设置冻结；原 PyTorch 合同和合格 CANN 合同分开保存。
2. 新覆盖层在候选 preflight／evaluate 之前做自包含源码静态检查和候选运行时 ATen 检查。自动检查不是完整静态证明，也不冒称人工审查。
3. KG 已完整评测过的同一源码不重复提交以挑分；候选按相对 CANN 的完整调用几何平均加速比排序，并补验原 PyTorch 语义、特殊值和布局。
4. 新候选通过数值后仍标记为待 Codex 逐路径审查、实际 TTIR／npubin 和设备路径核验；核验完成前不增加正式交付数。
5. 已提交但终态未知的请求不重发，不在未知用卡状态下交接。已结束且有完整结果的任务可恢复；存在运行中工作区时不重复启动。

人工审查发现控制流或缓存错误时，将带源码SHA和原因的记录保存在批次 `review-rejections/<算子>/`。控制器保留原始测量，但将该源码排除出最佳交付选择；错误的高分不能阻挡已修复版本。新增提案若已完成数值、补测、人工审查和后端核验并超过0.8×，后续尚未启动的搜索直接保留它，不再重复计时同一源码。`test_selection.py` 覆盖这两项选择规则。

原地算子冷启动须将编译与设备提交分开：`warmup`只编译，然后恰好执行一次。不能先执行原地kernel，再读取launcher，并在缓存构造失败时重做原地计算。编译缓存还须覆盖kernel、block、dtype、device及实际特化参数；传入不同的constexpr槽不会改变已经编译的二进制。相关源码审查事实进入后续任务上下文。

初始 `hardswish` 搜索中，原生KG自动执行了最终复验。已有两份同条件有效测量按几何平均汇报，未选较高的一次。后续覆盖层缓存相同源码、合同和设置的评测，避免原生最终复验重新计时；缓存副本不当作独立实测。相关规则与原始测量身份由 `measurements.py` 核验。

## 独立成对采集与切换边界

本批首次标准KGS profiling遇到缺少 `runner-completed` 标记，未取得可用指标。`paired.py` 因此直接运行 `msprof op`，对严格Triton候选与冻结CANN oracle采集同一原始性能输入，复用现有 `compare_opprof.py` 分Cube／Vector组比较绝对值、比率和平均搬运粒度。原始失败保留；独立采集也失败时如实标记，不编造硬件证据。`driver.py --side pytorch` 在此入口只表示执行spec绑定的oracle，参考身份明确为CANN。

每个源码SHA只采集一次代表输入，以起点最慢的原始case为依据，不改形状或容差。指标均值和完整任务时间分开解释；各流水可重叠，不相加为耗时分解。源码、实际IR、原始CSV、比较结果和精简知识一同进入角色上下文。

初始控制器保存于批次 `method-initial/`。`boundary_upgrade.py` 只在前一项结束、卡服务空闲、下一个工作区尚未启动时加载新控制器与新上下文，保留边界记录和角色SHA；不热改运行中的KG工作区。

## 最佳源码收尾

最佳源码的 `review.json` 必须由Codex实际逐路径检查后签署，不能自动批量生成。队列在串行空闲边界核验已审查候选；全部搜索结束后，也可使用 `finalize.py` 补齐运行时、实际TTIR／npubin和设备trace。源码、合同、正式结果、补测及参考资格合并核验后才更新交付总表。

参考缺口单列：`uniform` 的 CANN 下界舍入问题、`expm1_` 的 CANN 小幅值精度问题，以及 `fractional_max_pool3d` 的原生接口缺失，不能靠修改参考语义或伪称原生接口解决。

缺少合格CANN参考的数值修复提案也可在同一串行队列执行，正式身份明确为`original_PyTorch_only`，只复用一次原合同测量并追加正确性检查，不计入CANN达标数。提案的源码、冻结签名、原请求及参考身份分别核验；控制器常驻时，在空闲边界使用当前工具的验证入口。

下一轮上下文会携带本算子已失败提案的源码SHA、正式成绩和具体补测失败，并引用已有离线运算顺序负证据。失败记录用于排除重复尝试，不自动进入成功方法库。只有已有候选完成全部数值、接口、人工和后端核验且相对合格CANN超过0.8×，才跳过后续未启动提案。

`round_out`与`special_round_out`的out接口不同：前者可省略或传None，后者要求位置或关键字out。`signature_gate.py`在设备请求前核对冻结签名；`out_api_probe.py`分别补测12项分配、参数与尺寸路径。检查器按SHA冻结并随结果保存；恢复旧结果时使用原检查器身份。接口不合格的高分不能进入交付。

## 宿主启动与舍入公式的接续实验

`host_bridge.cpp`与`host_bridge_out.cpp`只读取张量元数据、缓存编译计划并调用Triton生成的启动器；对应构建脚本使用单CPU，源码、二进制和构建记录分别绑定SHA。每次调用重新读取输入／输出地址和当前stream，不能缓存数据或把提交异常转为再次执行。非连续、非对齐、空输入和计划缺失仍走原Triton路径。宿主桥不放宽候选的设备实现限制；只有逐函数审查、完整合同、补测和实际后端核验通过后才能交付。

本次对`round_out`的KG候选进行公式审查后，增加了保留负零和大幅值整数的快速最近偶数舍入提案，并迁移到`special_round_out`。`test_round_math.py`独立验证全域FP16／BF16及FP32舍入边界；CPU数学检查不代替实际Triton编译和完整设备评测。未通过补测的正式高分不会进入达标数。

## 离线总表与故障恢复

`report.py` 合并冻结的16项交付和本轮已核验候选，保存正式测量、源码身份、实际IR和设备路径；未完成后端核验的候选继续列为待核验。`audit_local.py` 可在Windows或Linux核验同步实体与SHA，两者均不执行设备评测。

```bash
python tools/profiling/hgkda/strict_night/audit_local.py
```

`component_probe.py` 将宿主调用、返回后同步等待和独立设备trace分开记录。厂商trace的时间戳可能为字符串，按十进制数求任务区间并集；各项不能相加为精确耗时分解，也不作为正式排名。本轮首次解析失败的trace和日志原件保留；恢复只重新采集未落盘的宿主诊断样本，复用原trace。

控制器重启前须确认原进程已结束、当前工作区没有活跃worker、项目KGS空闲，再在同一卡锁内恢复；不重启共享服务。共享路径使用`night_common.py`，避免不同队列的同名`run.py`在恢复时导入错误。切换边界按内容SHA保存实际执行工具快照。
