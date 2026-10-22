# log10_、selu_ 成对诊断与优化

本批目标：把老师提供的成对硬件分析方法接入HG-KDA，沿着“候选与参考差异→源码/IR机制→修改→完整评测”交付有效优化，以完整调用加速比0.8×为目标。不是仅采集指标，也不把手工修改归为模型自动成果。

## 最终结果

本批已结束：`selu_` **0.790×**、`log10_` **0.737×**，各36/36正式测试、42/42扩展语义与12/12连续快路径全部通过。最终源码的KGS profiling和四组成对OpProf采集完成，48项证据核验通过。整组尚未达到0.8×；历史较高分及失败候选均保留，不能移作最终成绩。

先读[精简汇报](精简汇报.md)、[方法与机制](方法与机制.md)，最终源码及各版本身份见[交付清单](deliverables-final-v6/manifest.json)。`completion-state-final-v6.json`和`completion-audit.json`分别记录终态与离线核验。旧`deliverables/`是追加实验前的临时导出，最终使用`deliverables-final-v6/`。

## 直接复现交付源码

在已配置本仓库的910B宿主执行，复用固定容器、KGS和卡锁；每次提供新目录：

```bash
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
python3 "$p/tools/profiling/hgkda/reproduce-paired.py" \
  --output /data/hanle/ascend-optimization/runs/paired-reproduction-新编号
```

该命令运行两项源码的42＋12项检查及各一次36项KGS评测，不重启服务、不重新生成源码。需要后台运行时，将完整命令置于新tmux会话。首次部署环境见仓库总README。复现时间可能波动，不能只比较汇总ratio而忽略候选和PyTorch各自延迟。

从GitHub获取后，`raw-evidence/`中的tar.gz必须为Git LFS实体；原始长路径OpProf和二进制保存在该归档，短路径源码、结果与比较JSON可直接阅读。对应manifest记录原始路径、长度和SHA。Linux可解压到独立新目录核对；不要覆盖已有实验目录。

## 固定条件

- FlagGems：`d6a8eec473517a3d68157b208eb9c057eb1d4c50`；FlagTree：`0.6.0+ascend.gitc286cba6`；Triton：3.5.1；CANN：9.0.0。
- 使用原有`tle_yy`环境、物理卡7；项目卡锁串行执行，不升级编译器或替换已有服务。
- KGS：19659；计分为完整调用walltime，36项合同沿用`night-20261021-v3`，见plan中的逐文件SHA。
- 新批次每个最终版本默认一次完整正式评测。候选探索和诊断探针分别保存，不用它们替代正式成绩。

## 目录与步骤

1. `seeds/`：原log10 HG候选与SELU追加修复候选，原始字节及来源保留；SELU并非纯原生KG结果。
2. `catalog/`、`plan.json`：冻结原合同与实验设备、计时范围。
3. `preflight/`：扩展语义检查、独立调用探针、与PyTorch成对的OpProf原件。`paired-0`是首次名称适配失败的记录；有效新目录为`paired-v2-*`。失败和有效结果分开保留。
4. `research/`：真实后端源码摘录与IR等机制证据。
5. `candidates/`：诊断后新增候选，来源写清HG-KDA证据与Codex修改。先42项扩展语义，包含原地别名、stride、offset、高rank、特殊值和stream切换，再做KGS完整评测。
6. `results/`：候选语义结果与正式KGS请求、结果、逐case延迟。

采集脚本与控制器位于`tools/profiling/hgkda/`：`run-paired-preflight.py`、`collect-paired.py`、`paired-driver.py`、`run-paired-candidates.py`。HG-KDA方法在`skills/hgkda/references/性能指标/成对对照与机制定位.md`，事实比较工具为`skills/hgkda/scripts/compare_opprof.py`。

这些驱动使用`msprof op --aic-metrics=Roofline`生成原始硬件计数；一条命令可能包含多次replay，不称为一次执行采完。正式计分、独立调用探针、设备OpProf分别解释。此批暂不宣称建立数值Roofline模型。

## 已发现的问题

旧log10候选的42项检查中，初次3项在测试输入构建阶段触发CANN Copy的rank限制，不能算候选错误。修正夹具、以1维底座构建和观察后，9维非连续输入仍有3项真实失败：旧候选截断到8维，漏算一半元素。新增候选使用完整shape/stride寻址，必须实机通过后才交付。

1024个fp16元素的log10成对采集，PyTorch设备任务约2.18微秒、旧候选约1.98微秒，而独立完整调用探针约5.97与15.51微秒。该小输入优先检查启动路径，不将两次独立采集相减为精确host耗时。

4096×4096的fp16成对采集中，两侧报告的主存读取量相近，但旧候选MTE2指令约为参考的2.68倍；设备任务约41.66与24.92微秒。继续检查搬运粒度和任务组织，不能仅根据MTE活跃比例宣布HBM饱和。

实际后端launcher源码显示：Tensor参数会执行`data_ptr()`和`aclrtPointerGetAttributes`，整数地址参数直接取指针。第一版候选据此尝试传入本次输入的实际地址；没有缓存输入Tensor、输入地址或结果。这个版本相关改法需经语义门禁、微探针和KGS性能核验。

## 第一版完整评测

| 算子 | 新候选相对PyTorch | 扩展语义 | 正式KGS | 来源 |
|---|---:|---:|---:|---|
| log10_ | 0.802081× | 42/42，每项两次含stream切换 | 36/36，一次完整评测 | 成对硬件证据＋Codex启动路径与高rank修复 |
| selu_ | 0.768048× | 42/42，每项两次含stream切换 | 36/36，一次完整评测 | 将同一启动路径方法迁移至SELU，并保留独立表达式/分块 |

两项候选原始文件为`candidates/*-rawptr-v1.py`；原始请求和逐case延迟保存在`results/<算子>/codex-rawptr-v1/formal/`。log10本次汇总刚过0.8×，逐case参考存在波动，不能据此宣称已稳定达标；后续优先降低候选实际延迟，不靠参考偶发变慢选版本。

`run-paired-kg.py`已将6组成对事实冻结到新覆盖层，使用CC/DeepSeek进行追加优化。该组另有生成与复验结果，不能把第一版Codex修复归为其自动成果。`run-paired-verification.py`等待同一卡锁，在KG终态后导出有效交付，并验证指针机制、编译IR和优化后的成对计数。状态分别在`candidate-state-v1.json`、`kg-paired-state.json`、`verification-state.json`。

## 最终版本与失败记录

最终log10使用`profilemeta-v5`，SELU使用`profilemeta-v4`；来源与SHA见交付manifest。`profilemeta-v3`修复KGS profiler需要的`size()`参数元数据；v4加入SELU按dtype分块，v5改变log10中间精度；v6检验关闭多缓冲后的大分块，连续检查8/12，未采用。`rawptr-profilemeta-v2`是未运行占位，不属于评测成果。

`finish-paired.py`完成补充门禁、按候选实际延迟选源码、KGS原生采集、种子/最终IR导出和最终四组成对采集；`audit-paired.py`离线核对合同、源码、请求、附件与角色加载证据。源码转换需容器Python3.11，宿主Python3.7只执行队列控制。日志中的3.7转换失败发生在候选与请求生成之前，已保留并修复。

本批冻结角色之后新增的知识条目仅供后续批次使用，不回溯声称影响了既有CC生成。log10保留主agent结构化分析记录，SELU另有独立分析角色委派记录；两者新候选的原生profiling失败也继续保留，最终采集修复另记。
