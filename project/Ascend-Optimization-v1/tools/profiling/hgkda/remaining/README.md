# 剩余九项连续攻关

本批为 `remaining-nine-20261030`。固定 FlagGems `d6a8eec473517a3d68157b208eb9c057eb1d4c50`、FlagTree `0.6.0+ascend.gitc286cba6`、CANN 9.0.0，沿用前批四份 KGS 合同。目标是改善八项性能未达标实现，并调查一项求逆正确性差异。

目标依次为 `negative_`、`expm1_`、`log2_`、`fix`、`round_out`、`special_round_out`、`hardswish`、`fmod_tensor_`、`linalg_inv_ex`。历史优化源码作为起点，固定版 FlagGems 原件另存 `seeds/`，二者不混称。

## 方法

先完整复验起点。Triton 候选迁移单缓冲、每个 program 处理两个块等有效方法；Ascend C 候选采用来源可见的计算源码和 C++ 同 stream 任务队列启动。特殊布局、未对齐地址、带小数位的舍入继续使用已验证的 Triton 路径。自写 Ascend C 不等于直接调用 CANN 目标库，也不称为纯 Triton 优化。

`expm1_` 在接近零时保留短级数和有符号零；`fmod_tensor_` 保留除法、截断、乘法、减法的数值顺序。求逆分别试验正确舍入除法、倒数乘法与融合乘加，调查重复行矩阵和 CPU LU 参考的舍入差异；不放宽容差、不删除奇异输入。

新源码通过一次完整 KGS 及原扩展补测后才保留。Ascend C 新候选先运行 `check_native.py`：1024 元素特殊值、同规格对齐切换、数据地址更新、当前 stream、原地别名与输出独立性；早期已完成候选在选择阶段补齐同一门禁。超过 0.8×的最终源码再固定一次完整确认，两次成绩取几何平均，低分保留。

`fix` 的早期 C++ 内部分配输出版本发生隔离 worker 停滞，已保存 HTTP 失败及进程身份，服务自动恢复后继续；未重启共享服务。后续 `fix`、`hardswish` 改用 Python `new_empty` 分配输出，计算和启动仍由自写 Ascend C 与 C++ 完成。观察到调用路径差异可解决本次停滞，尚未据此断言锁或运行时内部的具体成因。对应旧提案记录为撤回，不反复提交。

## 运行

在 910B 宿主机项目目录执行，控制器使用本项目物理卡 7 的锁。设备任务串行，不重启共享服务。

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
PY=/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python
$PY tools/profiling/hgkda/remaining/run.py --phase all
```

已有 tmux 会话时先查看状态，不重复启动控制器。完整任务可恢复，已有请求而结果未知时停止自动重发。

```bash
tmux ls
cat operators/campaigns/remaining-nine-20261030/进展.md
```

阶段可选 `verify`、`proposals`、`search`、`finalize`、`evidence`。`--operators` 限制起点复验及搜索列表；排队提案仍按完整批次验证。

队列命令恢复已有任务，不会重新提交已完成请求。需要对最终最佳源码做一次新的独立复现时，使用新结果名：

```bash
$PY tools/profiling/hgkda/remaining/reproduce.py --operator fmod_tensor_ --run-name manual-check-01
```

该命令执行一次完整KGS、原扩展补测和对齐原生路径检查，不改变历史最佳。复现其他算子时替换名称，保留已知失败与原容差。

## 文件

- `prepare.py` 冻结合同、软件版本与起点；不覆盖旧批次。
- `make_triton.py` 生成 Triton 提案；`make_native.py` 等冻结自写 Ascend C 候选与二进制身份。
- `ascendc/`、`ascendc_host.cpp`、`build*.py` 保存计算源码、C++启动封装及单 CPU 构建方式。后续构建使用新目录，原候选附件不覆盖。
- `run.py` 复用既有 KG/KGS 正式评测、扩展语义门禁及有预算的 CC/DeepSeek 搜索。
- `check_native.py` 检查真实经过对齐原生路径的特殊值与动态调用语义。
- `build_pipeline.py` 使用双缓冲队列组织搬运依赖；`make_allocation.py` 生成 Python 分配输出的独立候选。
- `evidence.py` 采集同输入的起点、PyTorch、最佳源码设备路径，以及实际 Triton IR。独立 profiling 时间不替代正式加速比。
- `finalize_remaining.py` 固定最终源码、补测与完整确认记录。
- `report_remaining.py`、`audit_remaining.py`、`review_knowledge_remaining.py` 核验最终交付、请求终态及知识复用资格。

实际结果位于 `operators/campaigns/remaining-nine-20261030/`。`进展.md` 是运行状态，`delivery.json` 与最终报告才是交付索引；队列启动和中途过线都不表示闭环。

Claude 完整运行日志可能含模型内部思考，不作为普通进展日志读取。查看 `*-controller.log`、原始评测 JSON、源码和已提交的 profiling 分析结论。
