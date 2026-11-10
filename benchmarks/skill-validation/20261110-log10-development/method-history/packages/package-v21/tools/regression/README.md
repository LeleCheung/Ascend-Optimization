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

## 回归后的修复实验

`unary_host_cache.cpp` 与 `build_unary_host.py` 用于验证 `negative_` 的启动开销假设。构建器保留父版本的全部 Triton 核和回退，追加只读取张量元数据并调用 Triton 启动器的 C++ 宿主缓存。缓存按元素数、类型和设备区分；每次取得当前地址和 stream，不保存输入张量或计算结果。

该构建器依赖冻结 `negative_` 的启动计划格式。生成内容属于独立新候选，须绑定宿主源码、二进制和父源码身份，经过逐路径审查、语义补测、实际 Triton 路径核验及一次完整 KGS 评测；它不改变原 25 项固定候选回归的结果。

`run_negative_repair.py` 等待固定回归、参考探测及随机参考检查结束，串行完成构建、准入、路径核验及正式评测。`verify_unary_host.py` 检查冷／热启动、新地址、新 stream、非连续和未对齐输入，捕获四类实际 Triton 编译核，检查宿主缓存是否命中。

`capture_compiled.py` 用于其他新候选的路径核验：在候选调用范围内拒绝非元数据 ATen 操作，并从实际返回的 Triton 编译对象保存 TTIR 和设备二进制。它与完整语义测试、依赖审查共同使用。

`compare_previous.py` 在相同源码和性能用例下分别计算候选耗时、CANN 耗时的新旧比值，输出逐项 JSON 和 CSV。历史索引含多次有效测量的汇总时，索引分数与所附单次原件分数分别保留；本轮仍每项只做一次正式测量。`run_tile_variant.py` 则独立核验 `hardswish_out` 的较大分块候选，所有结果与固定回归分开。

较大分块先检查实际编译路径。910B 的 UB 预算为 192 KiB；本轮 `hardswish_out` 的双循环版本在 8192 分块下仍然溢出，关闭自动多缓冲也未解决。`--complete-block-loop` 改为由静态整除条件选择单一循环，非整除保留掩码，避免完整块和尾块同时占用缓冲。数学表达式、非连续布局及输出类型回退不变。

最终组合还使用 `--affine-gate --native-low-precision --half-only-large-tiles`：归一化门控采用 `clamp(x/6+1/2,0,1)`，输入或输出为 FP32 时保留 FP32，其余采用输入精度。较大分块仅用于 FP16 输入和输出，BF16 内部扩展及 FP32 路径保留 4096。所有改动先核验实际编译产物及完整冻结补测，再进入一次正式请求；不放宽原容差。该组合为 0.866788×，30/30 及两侧各 42/42 通过。

`run_negative_repair.py --stream-variant` 继承经过核验的宿主缓存及 Triton 回退，仅将完整块改为无掩码固定长度搬运；尾块保留掩码。验证器的 `--include-stream-tail` 包含大输入非整除检查。编译异常后数值正确也不等于新路径生效，必须确认实际捕获目标流式核，之后才提交正式请求。
