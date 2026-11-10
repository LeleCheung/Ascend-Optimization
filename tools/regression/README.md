# 独立入口固定候选回归

本工具核验已有算子能否经独立 HG-KDA 入口完成相同评测，不调用 KG 的模型调度、优化循环或历史批次控制器。KGS 仍执行设备侧完整评测，补测使用冻结检查器；失败不改写旧验收记录。

## 准备

在安装了当前 HG-KDA 包的 Linux 环境中执行 `prepare.py`，显式指定维护仓库、历史归档、原项目路径和全新的运行目录：

```bash
python tools/regression/prepare.py \
  --repository /path/Ascend-Optimization \
  --archive-project /path/archive/project/Ascend-Optimization-v1 \
  --original-project /original/Ascend-Optimization/project/Ascend-Optimization-v1 \
  --run-directory /path/runs/new-regression
```

准备阶段按 `benchmarks/index.json` 选择已达标的 25 项，核对候选与审查哈希，原字节复制两侧合同及补测检查器，并核对原正式请求的合同包身份。历史 KG 快照仅保存 solution、identity 和 evaluation-contract 时，利用这些原件恢复相同来源、合同与计时参数的 HTTP 请求，并显式记录恢复方式。包含 CRLF 的候选保留原换行字节。

少量历史源码和参考以绝对路径加载已核验的 CANN 库或宿主启动桥。计划会枚举其文件身份；运行阶段只建立这些文件的临时兼容链接，正常结束后移除。候选的全部设备计算仍为 Triton，宿主启动桥仅承担分配、元数据和 Triton 启动。

## 执行

`run.py` 接收计划、共享容器名、项目分配设备、专用服务端口、设备锁、容器内 Python 和 CANN 环境脚本。应由 tmux 等独立会话执行，参数示例：

```bash
python tools/regression/run.py \
  --plan /path/runs/new-regression/plan.json \
  --container shared-container --device 7 --port 19663 \
  --lock /path/runs/physical-card-7-campaign.lock \
  --device-python /path/python3.11 \
  --cann-environment /path/cann/set_env.sh
```

控制器在项目锁内启动专用 KGS 服务，调用 `python -m hgkda gate`、`kgs upload` 和 `kgs evaluate`，再执行固定的 PyTorch／CANN 语义补测。每项仅提交一次正式请求；计时参数沿用原记录。输出包括源码、合同、请求、操作标识、原始结果、补测与逐项状态，不覆盖已有证据。

恢复时跳过已终结项目；已有评测目录但没有结果时禁止重发，须查询原 `operation_id`。异常结果未知时保留服务和依赖；处理原任务后再继续。已知设备失败、补测失败或性能不足单列，不追加复测挑分。控制器只停止带本批端口及产物目录身份的服务，不重启共享容器。

## 汇总

```bash
python tools/regression/summarize.py \
  --run-directory /path/runs/new-regression \
  --output /path/runs/new-regression/summary-final.json
```

生成独立 JSON、中文表格和逐 case CSV，包含候选耗时、CANN 耗时及加速比。新结果需同时满足完整正确性、严格 Triton 审查、两侧原语义补测及相对合格 CANN 参考超过 0.8×；不因脚本成功退出就认定全部达标。旧、新分数分别列出，两个汇总分数的比值不能直接解释为候选实际提速。

队列结束后执行原件核验：

```bash
python tools/regression/audit.py \
  --run-directory /path/runs/new-regression \
  --output /path/runs/new-regression/audit-final.json
```

核验候选与审查 SHA、原合同包、实际请求、计时参数、完整 case 集合、两侧冻结检查器和补测数量，并按原始耗时重算逐 case 比值及总分。`evidence_verified` 表示证据一致，`accepted` 表示满足本轮验收，性能未过线的完整证据不会被删除。

已归档批次可在 Windows 或其他 Linux 主机离线核验，不需要启动 KGS 或设备。`--original-run-directory` 只在内存映射原路径，`--verify-transfer` 核验传输清单中全部文件的长度和 SHA，原请求及合同不会改写。例如从仓库根目录执行：

```bash
python tools/regression/audit.py \
  --run-directory benchmarks/regressions/20261109/fixed-25 \
  --original-run-directory /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3 \
  --verify-transfer --output ../hgkda-output/new-archive-audit.json
```

输出路径须不存在且父目录已建立。使用普通克隆时先执行 `git lfs pull`，确保核验的是原始合同包及设备二进制，而非 LFS 指针。

## 专项与历史修复实验

通用回归工具不直接生成或改写候选。当前专项正确性／执行路径检查集中在[tools/operators/](../operators/README.md)，例如`negative_/verify_unary_host.py`；参考构建与资格检查也按算子维护。

2026-11-09的negative、hardswish_out及累计积分修复包含直接拼接或替换候选的历史控制逻辑，已退出当前运行入口，原字节保留于[tools/archive/20261109/](../archive/README.md)。已有成绩和失败原件不改写，详见[回归报告](../../docs/regression-20261109.md)。后续候选及启动封装均由CC／DeepSeek修改。

`capture_compiled.py`仍用于只读核验候选路径：拒绝候选调用范围内非元数据ATen操作，从实际Triton编译对象保存TTIR和设备二进制，与完整语义测试及审查共同使用。

`compare_previous.py`按相同源码和性能case分别比较候选与CANN的新旧耗时。它不以两个汇总加速比分数之比替代实际性能变化，不通过重复测量挑高分。
