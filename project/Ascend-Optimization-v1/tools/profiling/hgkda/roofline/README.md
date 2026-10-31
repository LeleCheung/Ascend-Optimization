# HG-KDA 数值 Roofline 模块

复用导师提供的通用分析器，把同次真机采集的计算量、各层流量和执行时间整理为可追溯的数值证据，再交给KG分析角色结合源码与IR提出实验。完整性能仍通过KGS评测。

## 各文件职责

| 入口 | 作用 |
|---|---|
| teacher_roofline.py、teacher-source.json | 导师Markdown代码的原逻辑提取及来源SHA |
| driver.py、calibrate.py | 受控GEMM与persistent copy计数校准 |
| empirical.py、empirical_audit.py | 同精度经验计算参考、大数组复制吞吐；后者保全原CSV并重算 |
| matrix_driver.py、collect_matrix.py | 父版／候选／PyTorch同输入目标kernel发现和采集 |
| profile_data.py | 读取官方可视化分段，校验设备／时间／kernel，保存组件流量与流水分布 |
| analyze.py、summarize.py | manifest、points、教师分析器输出及可注入角色的紧凑摘要 |
| candidate_template.py、search.py | 受控单输出tile候选、预算登记、编译／正确性／交错设备计时 |
| review.py | CC＋DeepSeek实际消费带数值证据的KG角色，保存提示、回执及provider记录 |
| finish.py | 追加实验、补充门禁、一次完整KGS、同case延迟选版和最终采集 |
| plot_chinese.py、audit.py、test_evidence.py | 中文图表、完整证据审计、真实归档与破坏性反例校验 |

## 数据口径

- 算法FLOPs、官方内部操作数分别报告。当前固定8192乘指令数的CSV不能统一乘128；FP32也不能借用FP16的计算参考。
- GM使用官方cache miss／eviction事件模型，L2和Vector等使用各自组件口径。Memory.csv主存字段、逻辑复制字节与GM计数字节不混用。
- GM经验参考使用同计数模型的persistent copy字节除以同次设备时间；大数组torch copy逻辑吞吐另列。两者都不是已证明的HBM理论峰值。
- 官方模型原值保留；本机0.8TB/s常量的匹配尚待核实。低于参考线不直接证明计算受限或访存受限，limiting_side只表示参考线较低的一侧。
- 纯复制没有有意义的算法FLOPs，不作为GEMM Roofline点；零计数、身份不符和非有限值拒绝构点。
- PyTorch当前只采GEMM组件，候选为融合kernel；图不能替代完整KGS加速比。

## 离线分析与接入

在仓库的 project/Ascend-Optimization-v1 下运行，批次路径以本轮为例：

~~~bash
batch=operators/campaigns/roofline-20261027
python tools/profiling/hgkda/roofline/analyze.py \
  --batch "$batch" --profile-directories matrix-profile candidate-profile \
  --output "$batch/新的分析目录" --plot
python tools/profiling/hgkda/roofline/summarize.py \
  --analysis "$batch/新的分析目录/diagnosis.json" --output "$batch/新的摘要.json"
python tools/profiling/hgkda/build-kg-overlay.py \
  --output /data/hanle/ascend-optimization/runtime/新的覆盖层 \
  --roofline-evidence "$batch/新的摘要.json"
~~~

分析输出目录已存在时拒绝覆盖。Windows长路径支持由分析入口处理；同步原件时使用传输清单逐字节核验。离线分析不占NPU，真机采集须先确认项目KGS空闲并持卡7互斥锁。

2026-10-27批次的结果与复现入口见[本批README](../../../../operators/campaigns/roofline-20261027/README.md)。本轮实际调用为冻结KG分析角色的单轮CC消费验证，不冒充重新运行完整KG自主搜索。
