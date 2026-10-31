# 矩阵乘片内复用与流水优化

针对 matmul_bias_activation 保留多个小累加块，改变复用方向与有限流水配置。输入是冻结父版、Roofline证据与候选空间；输出是通过语义与完整KGS的源码、同case真实延迟、同输入硬件流量和中文报告。

本批已完成：正式15项延迟比几何平均1.040×父版，相对PyTorch0.895×，42/42及18＋24项补充检查通过。冻结后相同模板仅改变复用方向，四项平均1.059×。结果及范围见[精简汇报](../../../../operators/campaigns/matmul-pipeline-20261027/精简汇报.md)。

## 方法

1. 从源码明确每个program的输出矩形及A／B逻辑请求，结合已采集GM／L2／L1流量选择变换。逻辑请求用于解释模板，不代替硬件字节。
2. 固定7个结构／缓冲候选，先在1024³ FP16筛选，将前两项扩展到1024／2048／4096与FP16／BF16。两次计时反转父版／候选顺序，保全task CSV。
3. CC＋DeepSeek读取冻结数值证据、源码、IR事实与实际搜索快照，从剩余3个预登记流水选项中最多选2项；相同规则验证，不自行扩大预算。
4. 所有完整开发集通过者按实际设备延迟选版。冻结后通过18项既有语义检查和24项大连续／矩形／非整除／stride／stream检查，再对父版与候选各做一次42项原合同KGS。
5. 重新采集父版与候选同输入profiling，核对流量与IR是否支持预测。新源码实际更快才采用，否则保留父版。用独立脚本核验选版、实际上传源码、task CSV、逐case延迟与模型输入。

## 文件

| 文件 | 职责 |
|---|---|
| templates.py | 多输出通用源码生成、配置校验和逻辑请求模型 |
| search.py | 有限搜索、开发门禁、交错设备计时与编译原件 |
| review.py | 实际CC／DeepSeek调用及冻结提示、回执和provider证据 |
| check_large.py | 24项补充输入与stream检查 |
| finish.py | 串行追加搜索、冻结选版、KGS、同输入profiling |
| analyze.py | 同输入各层流量、流水分布和IR的机制对照 |
| confirm_reuse.py | 冻结后相同模板的M／N复用方向机制确认，不参与重新选版 |
| audit.py | 真实原CSV、源码、合同、选版和模型输入独立核验 |
| report.py | 中文报告和逐case表格 |
| test_templates.py | 网格覆盖、逻辑请求与生成边界离线测试 |
| test_evidence.py | 正式15项结果、确认范围、硬件身份、模型输入与失败方案回归 |

## 使用

在 project/Ascend-Optimization-v1 下运行；所有输出均使用新目录，禁止覆盖历史。

~~~bash
/usr/local/python3.11.15/bin/python3.11 tools/profiling/hgkda/matmul_pipeline/search.py \
  --parent operators/campaigns/roofline-20261027/validation/delivered.py \
  --output /data/hanle/ascend-optimization/runs/新的矩阵开发目录 \
  --configs dual-n-constexpr dual-m-reuse-b quad-64x128 quad-128x64
~~~

真机阶段要求固定CANN／FlagTree环境，显式ASCEND_RT_VISIBLE_DEVICES=7、项目PYTHONPATH，且调用者持物理卡7项目锁。先检查19655／19657／19659全部健康空闲，不改共享服务、镜像或安装依赖。finish.py只在同一预登记批次、初轮搜索和模型回执完整后运行；失败停止且保留证据，不重发已完成正式评测。

本轮父版包含历史优化与共享兼容修复。CC实际选择追加配置，Codex实现模板、编排和最终核验；不把整批成绩归为模型自主生成。模型推断必须以源码与采集核对，BF16大K满足门禁时也进入新路径。

结果见[本批入口](../../../../operators/campaigns/matmul-pipeline-20261027/README.md)。

离线核验不调用NPU。在仓库根目录运行以下命令；核验输出须使用未存在的文件名。

~~~bash
python -m unittest discover -s project/Ascend-Optimization-v1/tools/profiling/hgkda/matmul_pipeline -p 'test_*.py' -v
python project/Ascend-Optimization-v1/tools/profiling/hgkda/matmul_pipeline/audit.py \
  --batch project/Ascend-Optimization-v1/operators/campaigns/matmul-pipeline-20261027 \
  --output /tmp/新的矩阵证据核验.json
~~~
