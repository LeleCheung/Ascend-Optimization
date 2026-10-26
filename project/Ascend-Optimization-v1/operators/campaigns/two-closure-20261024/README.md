# 两算子持续优化与交付（2026-10-24）

本批接续`remaining-five-20261023-v2`，只推进`upsample_bilinear2d`与`cumulative_trapezoid`。结论见[精简汇报](精简汇报.md)；默认参考和官方确定性条件分别记账。

## 最终结果

| 对象 | 条件 | 相对PyTorch | 完整测试 | 候选延迟几何均值 |
|---|---|---:|---:|---:|
| 上采样 | 默认原合同 | 0.915× | 24/24 | 0.100135 ms |
| 累计积分 | 官方确定性条件 | 0.812× | 33/33 | 1.730013 ms |

本批父版上采样为0.344894×，不是固定上游FlagGems性能。固定FlagGems本机原版存在正式编译／布局失败，不填造原版性能列。旧历史值继续保留在旧批次，不与本批平均。

累计积分默认参考存在同输入重复结果原容差区间无交集的问题，不能用偶然完整通过证明稳定可交付；[参考稳定性与复现](参考稳定性与复现.md)保留反例、输入SHA和独立重现方法。本批确定性结果是明确的新执行条件，未静默替换默认参考。

## 固定环境与边界

- FlagGems：`d6a8eec473517a3d68157b208eb9c057eb1d4c50`。
- FlagTree：`0.6.0+ascend.gitc286cba6`；Triton3.5.1；CANN9.0；torch2.9.0+cpu；torch_npu2.9.0.post2。
- 只使用910B物理卡7，容器`tle_yy`、逻辑`npu:0`、KGS19659。
- 全部设备实验串行持锁`/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock`；不重启共享服务，不修改共享后端，不操作他人进程。
- KGS完整调用walltime，包含host与device，三次trial；一份新源码一次正式独立评测。KG内部搜索测量仍原样保留，不据此挑重复高分。

## 如何复现

1. 将仓库克隆到910B并执行`git lfs pull`，按照根README恢复锁定环境与独立KGS19659。确认无其他本项目卡7评测在运行。
2. 进入910B宿主仓库项目目录：

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
```

3. 用新的输出标签复现上采样最终源码，工具自动持卡锁并核验精确源码SHA：

```bash
python3 tools/profiling/hgkda/reproduce-two-delivery.py \
  --operator upsample_bilinear2d --label upsample-new-reproduction
```

4. 累计积分必须显式选择官方确定性条件；参考、容差和KGS引擎保持原样：

```bash
python3 tools/profiling/hgkda/reproduce-two-delivery.py \
  --operator cumulative_trapezoid --deterministic --label scan-new-reproduction
```

5. 结果位于本批`reproductions/<标签>/`，包含请求、条件、准入、补充检查、完整结果和KGS作业回执。标签不能重复。该入口仅复现最终源码，不重新启动模型搜索。
6. 默认参考波动复现见独立报告。需要重新搜索时新建批次，不修改本批合同、知识快照或失败证据。

## 文件索引

- `deliverables/reviewed/`：两份最终源码；精确SHA见`final-delivery.json`。
- `catalog/`、`seeds/`、`plan.json`：原合同、本批起点、搜索预算。
- `knowledge/`：按实验时间保留的提示、探针、候选和失败机制；早期“尚未完成”描述是当时快照，当前结论以最终清单为准。
- `results/`：默认合同的KG搜索与独立评测原件；`codex-disjoint`是最终上采样修复版。
- `research/`：默认参考稳定性、官方确定性评测、源码IR、真实路径、独立控制记录。
- `evidence-audit.json`：合同、源码、完整测试、逐case和机制证据核验。
- `raw-evidence/`：原始附件LFS压缩归档，下载与逐文件SHA清单由同步工具生成。

## 计分与归属

按同case候选实际延迟选版，不能用不同运行的相对PyTorch分数直接排序。连续读取扫描版0.782×与FIFO版0.770×分数高低并不说明延迟排序；最终采用的启动优化版为0.812×，实际几何延迟1.730013ms。上采样CC原候选的0.934×及后续复验原件保留；因交界写入重叠不直接交付，采用通过修复、完整数值和唯一归属检查的版本。

KG运行模式、Codex诊断修复、CC/DeepSeek生成分别记账。两次预算取消后的worker完成元数据、空闲确认与恢复证据保留；不把旧`bootstrap-state.json`的历史needs_attention当作最终状态。当前入口以`final-delivery.json`为准。
