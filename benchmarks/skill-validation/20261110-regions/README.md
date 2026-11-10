# 整理后入口复验与区域方法证据

`entry-regression/`保存SELU、log10固定源码的一次完整复验；`log10-regions/`保存由既有生成候选出发的两项CC／DeepSeek修改、方法快照、模型调用及全部正式／采集结果。详细结论见[中文报告](../../../docs/regions-method-20261110.md)。

| 证据 | 内容 |
| --- | --- |
| [入口核验](verification/entry-regression.json) | 57份导出文件、两项完整结果及补测 |
| [方法核验](verification/method-run.json) | 2750份导出文件、两份新源码的完整结果 |
| [生成轨迹](log10-regions/generation-audit.json) | 60文件方法身份、3次调用、候选Edit重放及方法工具读取 |
| [预期与实测](log10-regions/method-validation.json) | 各版本性能、真机任务、搬运粒度、单核时间轴及保留决策 |
| [时间轴复算](verification/regions-replay.json) | 三份时间轴与区域重算、单位核验 |
| [失败知识检索](verification/knowledge-replay.json) | 基于同源码完整成绩的失败反例准入与检索 |
| [26项既有交付核验](verification/current-26.json) | 四批1159条传输记录、28条正式记录汇成26项交付；历史关联文件另存回执 |
| [Windows安装回执](verification/installation-windows.json)、[910B安装回执](verification/installation-910b.json) | 同一wheel及独立环境中的CLI入口验证；源码版本9928d881f |

原始Linux绝对路径保留，核验时使用`dependency-map.json`显式映射已有固定回归依赖；不修改原件。

从仓库根目录执行：

```bash
python tools/skill_validation/verify.py --directory benchmarks/skill-validation/20261110-regions/log10-regions --repository . --output /tmp/regions-method-check.json
python tools/regression/audit.py --run-directory benchmarks/skill-validation/20261110-regions/entry-regression --original-run-directory /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression --dependency-map benchmarks/skill-validation/20261110-regions/dependency-map.json --verify-transfer --output /tmp/regions-entry-check.json
```

输出必须使用新路径。Windows将`/tmp/`换成仓库外目录。模拟器超时、单位核验失败及应用退出异常日志均保留，接纳范围见报告；缓存和凭据不在归档内。
