# HG-KDA 分阶段体检与 KG 接入

本目录对应 2026-10-15 会议后的下一阶段。固定版本见[基线锁定](../../../../../manifests/hgkda-stage2.lock.json)，15 个对象的登记见 [targets.json](targets.json) 和[下一阶段计划](../../../operators/下一阶段计划-20261015.md)。

## 方法与实现

入口 skill 是 [skills/hgkda/SKILL.md](skills/hgkda/SKILL.md)。知识同时按算子类型和性能指标分类，路由见[知识索引](skills/hgkda/references/知识索引.md)。

`scripts/diagnose.py` 从逐 case 数据产生瓶颈假设、知识路由与下一阶段采集建议。简单算子先用 metrics；复杂路径或多流水活跃难以解释时，升级 instruction 并检查同步与热点。当前不是定量 component-based Roofline。

`../hgkda_adapter.py` 连接现有控制器与 KGS：每轮最多新增一次代表 case 的指令采集，校验附件，按候选源码中的 jit 符号过滤并提取模拟器热点。原始附件保留，模型获得结构化事实和分类知识。缺少同候选证据、后端不支持或请求结果未知时，不假装完成诊断。

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

## 固定基线与验证

```bash
python tools/profiling/hgkda/verify-baseline.py \
  --server http://127.0.0.1:19655 --output /新批次/baseline-verified.json
```

核对固定 FlagGems 的逐文件 SHA 与 KGS 报告的 FlagTree、Triton、CANN、驱动版本。PR #1238 的编译器变更另设实验组，不能在同一组对照期间升级。

离线测试见 `tests/test_diagnosis.py`，验证缺失指标、非法比例、重叠流水升级、服务能力限制及真实历史指令的候选热点筛选。完整正确性与性能必须由 910B 的 KGS 验证。
