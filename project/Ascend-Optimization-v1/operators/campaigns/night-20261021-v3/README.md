# 固定版本的15项体检与KG四组对照

本批对应2026-10-15会议指定的10项算子及5项FlagGems-sglang算子。两个重点为 `value_selecting_reduction_backward`、`moe_sum_reduce`，各比较固定原版、KG无profiler、KG原生profiler、HG-KDA接入KG。另运行 `selu_`、`log10_` 的HG-KDA组。

全部原始组、六份独立修复、SELU追加实验和启动审计均已结束，实验交付核验通过。最新成绩见[精简汇报](精简汇报.md)，有效源码见[交付清单](交付清单.md)。原始组的扩展语义失败保留在 `结果汇总.md`、`统计核验.json`、`finalization.json`，修复分别核验；队列的 `passed` 只表示对应步骤通过。

## 固定环境和计分

- FlagGems：`d6a8eec473517a3d68157b208eb9c057eb1d4c50`。
- FlagGems-sglang：`938612058a981487e439da514df7d1371cf31e68`，本批保存选定实现、测试和benchmark的上游快照。
- FlagTree：`c286cba6d5c30f07988c0414beb21fd2be1abb3e`；Triton 3.5.1、CANN 9.0.0。
- 910B物理卡7，`tle_yy` 容器内逻辑 `npu:0`；独立KGS端口19659。
- 正式分数为完整候选调用walltime，包含host和device；profiling另采设备任务与流水指标。历史设备kernel计时单独报告。
- 每个生成组2个正式轮次、1个Coder会话；debug jobs和采集另计。各组起点相同，首轮生成代码不自动算作原版。

固定原版正确性通过后才计算其性能。`fractional_max_pool3d` 和 `conj_physical` 的参考包含CPU回退，保留原始数据，单列解释；不将这些聚合分数作为相对原生PyTorch NPU的加速成果。

导出的FlagGems快照没有构建时生成的 `_version.py`，其 `__version__` 可回退读取已安装包的metadata；日志里该字符串不等同于当前导入源码的提交。固定版本以实际源码路径、锁定清单和逐文件SHA核验为准。

## 文件入口

| 位置 | 内容 |
|---|---|
| `plan.json` | 固定源码、合同、用例数量及SHA |
| `catalog/ops/<算子>/` | Native定义、正确性/性能用例及独立参考 |
| `seeds/` | 本批固定原版入口 |
| `results/<算子>/<模式>/export/main.py` | 从该组完整通过轮次导出的源码 |
| `results/<算子>/<模式>/independent/` | 同一源码两次完整复验，含请求、bundle和逐case结果 |
| `results/<算子>/<模式>/native-evidence/` | 原生KG角色调用、各轮源码、采集、分析与debug jobs归档 |
| `results/<算子>/<模式>/supplementary-semantics.json` | 不参与性能计分的未见形状、stride和参数补测 |
| `results/<算子>/版本对比.*` | 按同case延迟计算相对PyTorch和固定FlagGems的比较 |
| `research-observations/` | 上下文、有效设备计数及冻结覆盖层的独立核验 |
| `reference-*-audit.json` | 参考执行路径与CPU回退记录 |
| `deliverables/<算子>/<源码SHA>/` | 已通过扩展语义及两次完整复验的源码，以及逐case对比 |
| `交付清单.*`、`completion-audit.json` | 最终选择与实验交付核验；Git三端同步另行核验 |

三个接口适配、四个value额外候选、两个MoE修复及SELU追加实验在同级 `night-20261021-v3-*` 批次独立保存。旧`postflight.json`不包含后来增加的全部批次；完整收尾以`delivery-finalization.json`、`all-repairs-verification.json`、`selu-repair-verification.json`和`completion-audit.json`为准。额外修复的预算、父源码及分析来源单列。

## 复测已保存源码

此操作复测已保存的实现，不重新调用模型生成。等待串行队列与后置任务结束，确认服务空闲；使用新的输出目录。示例复测已通过36项扩展语义的MoE修复版，执行8项正式合同及两次完整调用性能：

```bash
root=/data/hanle/ascend-optimization
p="$root/Ascend-Optimization/project/Ascend-Optimization-v1"
batch="$p/operators/campaigns/night-20261021-v3"
out="$root/runs/retest-moe-新批次"

flock -n "$root/runs/physical-card-7-campaign.lock" \
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=7 \
  -e NO_PROXY=127.0.0.1,localhost -e no_proxy=127.0.0.1,localhost \
  -e "PYTHONPATH=$root/Ascend-Optimization/third_party/FlagGems/src:$root/Ascend-Optimization/sources/kernelgen_server/client:$root/Ascend-Optimization/sources/kernelgen_server:$root/Ascend-Optimization/sources" \
  tle_yy bash -c 'source /usr/local/Ascend/cann-9.0.0/set_env.sh && exec "$@"' bash \
  /usr/local/python3.11.15/bin/python3.11 \
  "$p/tools/evaluation/evaluate-operator.py" \
  --operator moe_sum_reduce --operator-dir "$batch/catalog/ops/moe_sum_reduce" \
  --source "$batch/deliverables/moe_sum_reduce/366b70f0b93208bac6353567f71d72f705f11cc1b93066ce5f3c2040dc595c5f/main.py" \
  --label HG-KDA-retest --output "$out" \
  --server http://127.0.0.1:19659 --repeats 2 --timing-scope walltime
```

正式复现需核对provenance中的实际源码SHA、合同及全部逐case结果。重新执行一般语义检查时，在相同容器、CANN环境和卡锁下运行`tools/profiling/hgkda/check-nightly-semantics.py --operator moe_sum_reduce --source <同一交付源码> --output <新目录>/supplementary-semantics.json`。value使用同一检查器并指定其算子名；其正式合同为96项。

## 观察、恢复与新生成

Windows/Linux控制端在总仓库执行：

```bash
python project/Ascend-Optimization-v1/tools/control/nightly-status.py
python project/Ascend-Optimization-v1/tools/control/fetch-nightly.py \
  --batch night-20261021-v3 --snapshot
```

运行中仅取临时快照；终态后去掉 `--snapshot` 正式导入。服务环境、tmux、运行工作区不要求复制到Git仓库。

需要恢复时先核实当前tmux、worker及KGS请求终态，再使用原plan和相同批次的 `run-nightly.py`；请求结果未知时不能重复提交。新生成实验使用新的批次目录和覆盖层，按[HG-KDA入口](../../../tools/profiling/hgkda/README.md)执行。当前仓库知识已补充本轮经验，重新生成不等同于原始冻结组；须记录所用覆盖层manifest、模型与预算。
