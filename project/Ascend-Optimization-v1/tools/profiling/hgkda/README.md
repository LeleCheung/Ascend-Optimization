# HG-KDA 分阶段体检与 KG 接入

本目录对应 2026-10-15 会议后的下一阶段。固定版本见[基线锁定](../../../../../manifests/hgkda-stage2.lock.json)，15 个对象的登记见 [targets.json](targets.json) 和[下一阶段计划](../../../operators/下一阶段计划-20261015.md)。

最新[缓冲与搬运批次](../../../operators/campaigns/buffer-pointwise-20261022/README.md)已完成：log10_ 0.875×，较同批父版实际加速1.125×；selu_保留旧源码，本批复测0.898×。36项正式测试、42＋12＋9项补充检查和48项证据核验通过。新增缓冲生命周期知识、数值范围门禁、异常恢复及同字节评测复用；CC生成与Codex整合分别记录。复现使用`reproduce-paired.py --batch buffer-pointwise-20261022 --tag final --output /data/hanle/ascend-optimization/runs/新目录`，模拟器失败与最终真机采集分开报告。

2026-10-22新目标：[log10_、selu_成对诊断与优化](../../../operators/campaigns/paired-pointwise-20261022/README.md)。新增同输入成对采集与设备计数比较，结合实际launcher/IR提出修改。`build-kg-overlay.py --paired-evidence 摘要.json --native-context`可把冻结seed的成对证据注入新KG分析角色；角色明确区分起点证据和新候选测量。新批次最终默认一次完整KGS评测，历史批次保留原验证次数。

该目标本批已完成闭环：log10_最终0.737×、selu_最终0.790×；各36/36、42＋12补充检查通过，KGS原生profiling与四组成对采集成功，48项证据核验通过。成果包含当前地址启动、log10中间精度优化、SELU按dtype分块，仍未达到整组0.8×。`finish-paired.py`串行收尾，`audit-paired.py`离线核验证据，`reproduce-paired.py --output /data/hanle/ascend-optimization/runs/新目录`独立复现交付源码。快路径依赖固定FlagTree的私有launcher ABI，换版本须重新验证。冻结覆盖层未回溯应用后续知识更新。

2026-10-21 正在执行[夜间大goal](../../../operators/夜间大goal-20261021.md)：`prepare-nightly.py` 构建冻结的Native合同，`run-nightly.py` 串行体检并运行原生KG，`finalize-nightly.py` 等待卡锁后补充语义检查与统计核验。`report-nightly.py` 检查实际上传bundle、源码、用例、两次完整复验和profiling采集/分析证据。控制端用 `../../control/nightly-status.py` 查询；仅启动队列或加载角色不代表完成优化。

本批已观察到原生KG主动从metrics升级到CANN模拟器；首次instruction请求超时，成功metrics与失败采集分别保留。配置、预算及后续方法问题见[指令采集观察](research/原生KG指令采集预算-20261021.md)。该记录不改变运行中的冻结覆盖层。

## 方法与实现

入口 skill 是 [skills/hgkda/SKILL.md](skills/hgkda/SKILL.md)。知识同时按算子类型和性能指标分类，路由见[知识索引](skills/hgkda/references/知识索引.md)。

`skills/hgkda/scripts/diagnose.py` 从逐 case 数据产生瓶颈假设、知识路由与下一阶段采集建议。简单算子先用 metrics；复杂路径或多流水活跃难以解释时，升级 instruction 并检查同步与热点。当前不是定量 component-based Roofline。

`../hgkda_adapter.py` 连接现有控制器与 KGS：每轮最多新增一次代表 case 的指令采集，校验附件，按候选源码中的 jit 符号过滤并提取模拟器热点。原始附件保留，模型获得结构化事实和分类知识。缺少同候选证据、后端不支持或请求结果未知时，不假装完成诊断。

新批次的指令目录使用 case_id 的摘要命名，支持 Windows/Linux 同步。2026-10-15 首批原始证据中的含冒号路径已仅改文件名，内容不变，映射见该批的 `evidence-path-relocations.json`；原始 JSON 与提示词中的历史路径依此映射查找。

## 使用现有控制器

仅在新的批次给 `optimization_campaign.py` 增加 `--hgkda`，继续使用原 KGS 完整评测、最佳保留和两次复验。默认不带此参数时沿用旧流程，历史结果不重新解释。

```bash
python tools/profiling/optimization_campaign.py \
  --operator rad2deg_ --seed operators/rad2deg_/candidates/flaggems-fixed.py \
  --output operators/rad2deg_/reports/hgkda-新批次 \
  --knowledge tools/profiling/hgkda/阶段通用知识.md \
  --rounds 2 --plain-json --provider-thinking disabled --hgkda
```

控制器将全部 skill 文件摘要写入新批次配置，恢复时知识改变会停止。采集选择与模型预算须随结果报告，不能把已有知识指导的结果声称为无先验自动发现。

`rad2deg_` 新候选在正式评测前自动执行21项补充语义检查，并核对源码 SHA。它覆盖当前正式合同之外的原地别名、非连续布局和奇数长度；其他算子尚未接入这项专用门禁。首批真实结果、修复贡献与未采纳指令候选见[首项交付](../../../operators/会议后首项交付-20261015.md)。

## 插入 KG 原生分析流程

`build-kg-overlay.py` 将固定 KG 生成到独立目录，只在两种原生 profiling 分析角色后追加 HG-KDA 方法与分类知识，并附带标准 skill。KG 原有 MCP 工具、证据登记合同、Coder 和 KGS 评测继续使用。原生角色仍由模型推理和调用采集工具；确定性规则脚本的自动执行由上面的控制器适配器负责，两种模式分别记录。

在 910B 宿主机或带符号链接权限的 Linux 环境生成：

```bash
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
python3 "$p/tools/profiling/hgkda/build-kg-overlay.py" \
  --output /data/hanle/ascend-optimization/runtime/hgkda-新批次/sources

export KG_SOURCE_ROOT=/data/hanle/ascend-optimization/runtime/hgkda-新批次/sources
bash "$p/tools/evaluation/run-kg-operator-910b.sh" \
  rad2deg_ profile hgkda-新批次 2 "$p/operators/rad2deg_/candidates/flaggems-fixed.py"
```

覆盖层记录原角色与扩展角色 SHA，源码根目录通过 `KG_SOURCE_ROOT` 显式选择。简单旧入口可能不自动启用 skill，但两种分析角色都已带上方法说明；不能把目录中存在 SKILL.md 当作模型实际调用的证据。新原生实验需核对工作区实际加载的角色和 MCP 分析登记。

后续新覆盖层可在构建命令中增加 `--native-context`：把该轮冻结的Native输入规格附加到 `get_profile_context` 的 `hgkda_workloads`，并校验候选、测量、逐case参数及profiling集合SHA。原有测量字段保持原值，避免模型在 `axes` 为空时猜错形状。已在真实MoE第二轮工作区完成只读接口验证，详见[MoE分块与计时诊断](research/MoE归约的分块与计时诊断-20261021.md)。本批 `night-20261021-v3` 的冻结覆盖层没有启用这项后续扩展，不把接口验证写成优化收益。

## 离线读取 OpProf 设备计数

`opprof_facts.py` 读取KGS保存的 `msprof-op-report.tar.gz` 和对应 `manifest.json`，验证附件SHA、候选solution SHA、case和kernel身份，输出逐block搬运/流水/冲突指标分布、报告的搬运总量及计数加权L2读命中率。缺失值保持缺失，比例不相加，原始附件不解压或修改。

使用Python 3.8及以上；910B宿主默认Python 3.7，请在 `tle_yy` 内使用 `/usr/local/python3.11.15/bin/python3.11`。此过程离线读取文件，不调用NPU。假设变量 `profile_dir` 指向归档中包含附件与manifest的目录：

```bash
python opprof_facts.py \
  --archive "$profile_dir/msprof-op-report.tar.gz" \
  --manifest "$profile_dir/manifest.json" \
  --expected-archive-sha256 957b7b956505fc4b29578384ade826ec6b85ad039941f32c883506fe71c41432 \
  --expected-solution-sha256 f9fff605fcdd834bae26008521efd331194da3e04f0d179d1caa34ee924d355d \
  --case-id sglang_moe_sum_reduce-00004-timing \
  --kernel _moe_sum_reduce_kernel --output /新目录/moe-opprof-facts.json
```

这些SHA与case参数仅属于本次MoE HG第二轮。其他候选须从各自原始identity/manifest和附件核对，输出路径必须未存在。当前真实样本的设备计数有效，但模拟器阶段超时，因此输出标记 `partial_collection=true`、`instruction_listing_available=false`；失败指令请求和有效设备采集分别报告。

表中的block记录不是物理核数量；逐block带宽usage不是全卡HBM峰值利用率，厂商memory接口计数也不自动等于物理HBM流量。该工具尚未自动接入本轮冻结的KG分析角色。真实结果和后续诊断见[MoE分块与计时诊断](research/MoE归约的分块与计时诊断-20261021.md#opprof逐block设备计数)，专项测试为 `tests/test_opprof_facts.py`。

分阶段诊断脚本现要求指令请求 `status=completed` 且存在 `instruction_listing`，才登记指令证据；只有日志、OpProf设备报告或失败回执时保留metrics阶段。适配器仍不会自动重提已有失败请求。对应回归测试已覆盖这次MoE“设备计数成功、模拟器失败”的情况；本次改动只供后续批次使用。

`selu_` 的真实指令、调用路径和分块探针见[逐元素启动诊断](research/逐元素算子的启动路径与分块诊断-20261021.md)。`audit-selu-launch.py` 是需要同一卡锁的额外NPU诊断，核对固定候选的缓存key、真实直接启动和计算，不作为正式性能成绩；不是离线OpProf解析的一部分。

## 夜间批次的最终源码交付

原主队列和补充语义任务结束后，在容器Python 3.11或控制端Python 3.8及以上运行：

```bash
python tools/profiling/hgkda/export-nightly-deliverables.py \
  --batch-dir operators/campaigns/night-20261021-v3
```

工具重新核验两个重点算子的冻结合同、源码身份、补充语义及两次完整复验；额外修复还核验父源码谱系。仅在这些检查通过后，按相同case的实际延迟选出有效最佳版本，将原始源码字节写入批次 `deliverables/<算子>/<源码SHA>/main.py`，生成 `交付清单.md` 和 `交付清单.json`，并调用现有比较工具输出该源码与固定FlagGems的逐case Markdown、CSV及JSON。尚有待完成候选或重点算子没有有效源码时返回非零。

排名使用固定FlagGems延迟与候选延迟之比，跨两次复验取几何平均，避免PyTorch参考耗时波动改变候选排名。原始相同预算实验和额外Codex修复分别标记；导出交付源码不改变原始四组报告的严格门禁或失败证据。

`run-semantic-repair.py` 还支持单独登记的 `moe_semantic_repair` 模式：保留原始父源码及36项语义失败快照，先通过全部36项、再执行两次原8项正式合同。其核验与原value修复共用 `verify-semantic-repairs.py`，算子、完整case集合、检查器、源码和设备身份均须对应。`moe-stride-repair` 和 `moe-hg-layout-repair` 属于额外Codex修复，不在旧postflight预置等待名单中，收尾时必须分别检查；交付导出工具已识别这两个批次。

## 固定基线与验证

```bash
python tools/profiling/hgkda/verify-baseline.py \
  --server http://127.0.0.1:19655 --output /新批次/baseline-verified.json
```

核对固定 FlagGems 的逐文件 SHA 与 KGS 报告的 FlagTree、Triton、CANN、驱动版本。PR #1238 的编译器变更另设实验组，不能在同一组对照期间升级。

离线测试见 `tests/test_diagnosis.py`，验证缺失指标、非法比例、重叠流水升级、服务能力限制及真实历史指令的候选热点筛选。完整正确性与性能必须由 910B 的 KGS 验证。
