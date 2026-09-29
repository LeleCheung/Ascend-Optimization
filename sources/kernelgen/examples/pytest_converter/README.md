# Pytest Converter

`pytest_converter` 将本地 `kernel_todo_v2/pytest_conversion_inventory.json` 中的 FlagGems 算子测试迁移到 `gems_op`、Workload 枚举和精确重放规范。`kernel_todo_v2/` 是 Git 忽略的实验状态目录，不随仓库分发；首次运行前必须准备 `20260826.csv` 和各芯片的 `failed_ops.txt`，再按下述命令生成 inventory。当前默认流程每批选择 30 个目标文件互不冲突的算子，在独立临时 worktree 中同时启动转换 Agent，再由主 Agent 集中审阅、落地和验证；串行流程保留用于单算子返工和协议边界排查。

## 并行批量流程

加载 Claude 配置后运行一批 30 个转换 Agent：

```bash
cd /data/akg_kernel_bench_lite/kernelgen
source env.opus.sh
export ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN"
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_MODEL="$MODEL"
PYTHONPATH=.. python3 examples/pytest_converter/run_batch.py --size 30
```

`run_batch.py` 只在临时 worktree 中并行生成补丁，不直接修改共享 FlagGems 工作树。主 Agent 必须逐项检查补丁，将确认后的修改落到共享工作树，并通过 `record_batch_review.py` 保存审阅后的文件快照。发现现有协议无法表达算子语义时，将其记录到 `kernel_todo_v2/pytest_conversion_protocol_gaps.md` 和状态文件；若修改前基线已在目标芯片失败，则记录到 `kernel_todo_v2/pytest_conversion_validation_blockers.md`。两类条目都等待统一确认并由后续批次自动跳过，不得为了通过流程强行改写测试。

自动选批还会跳过已有 `ready_for_review`、`reviewed` 或 `reviewed_with_fixes` 批次结果的算子，避免设备验证和正式批准尚未完成时重复启动转换 Agent；需要显式返工时使用 `--operators` 指定算子。

审阅完成后，默认在本地 `kernelgen-nvidia-cu128` 容器中并行验证整批。先用 `nvidia-smi` 确认真正空闲的设备，再通过 `--devices` 传入；验证器为每张卡启动一个 worker 从公共队列领取任务，保证每卡同时只执行一个算子。容器直接读取挂载在 `/data/akg_kernel_bench_lite/FlagGems-master` 的当前工作树，每个算子使用独立 `/tmp` 副本执行，不会在基线和转换后测试之间覆盖共享仓库。

```bash
docker exec kernelgen-nvidia-cu128 nvidia-smi
python3 examples/pytest_converter/validate_batch_nvidia.py --batch kernel_todo_v2/pytest_conversion_batches/<batch>/manifest.json --devices 0,1,2,3,4,5,6,7
```

`validate_batch_metax.py` 继续保留为沐曦历史结果复现入口，不再作为 pytest 转换的默认正式验收环境。

需要持续利用本地空闲 NVIDIA 卡消化已审阅队列时，可以运行 `python3 examples/pytest_converter/run_nvidia_backlog.py --devices <空闲卡列表>`。脚本只调度 `reviewed` 或 `reviewed_with_fixes` 且文件 hash 仍与审阅快照一致的算子，并在完整验证通过后依次完成串行批准；显存不足会等待空闲卡重试，语义失败会停止该算子。远端 PPU 入口为 `run_ppu_backlog.py`，连接、容器和模板目录必须通过命令行按 `tests/hosts.md` 显式传入，脚本不保存主机凭据或临时环境默认值。

已批准算子因共享 pytest 文件修正而需要重新验收时，运行 `python3 examples/pytest_converter/prepare_revalidation.py --operator <算子> --notes <原因>`。该命令保留原批准历史，追加最新的 `needs_changes` 决策并生成只包含当前文件快照的 revalidation batch；后续选择和验证必须以每个算子的最新决策为准。

上一批设备验证期间可以提前启动下一批转换 Agent，但不得落地会修改上一批目标文件的补丁。若两批算子共享 correctness 或 benchmark 文件，必须等上一批验证结束后再落地下一批修改，避免审阅快照与远端验证代码不一致。验证通过后仍需逐项执行 `open_batch_review.py` 和 `record_review.py --decision approved`，正式写入串行状态。

## 串行流程

1. 从最新 CSV 和 FlagGems checkout 重新生成全量清单：

```bash
cd /data/akg_kernel_bench_lite
PYTHONPATH=. python3 kernelgen/tools/build_pytest_conversion_inventory.py
```

2. 加载 Claude 配置并启动一个算子。`env.opus.sh` 可能包含凭据，只能 `source`，不得打印或写入日志：

```bash
cd /data/akg_kernel_bench_lite/kernelgen
source env.opus.sh
export ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN"
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_MODEL="$MODEL"
PYTHONPATH=.. python3 examples/pytest_converter/run_one.py --operator gt_.scalar
```

当前开发机的 Python 环境没有安装 Torch，正式串行迁移在这里统一加 `--no-tests`，避免 Agent 重复执行必然 collection 失败的命令；这只关闭 Agent 本地 pytest，不降低验收要求。主 Agent 审阅 scoped diff 后，仍须按规范在本地 `kernelgen-nvidia-cu128` 容器完成迁移前后正确性对比、完整 core benchmark 对比、Workload 枚举、preflight、精确重放、profiling 和 override 恢复检查，再记录 `approved`。

```bash
PYTHONPATH=.. python3 examples/pytest_converter/run_one.py --operator gt_.scalar --no-tests
```

3. 人工审阅 FlagGems 中该算子的 scoped diff、Agent JSON 和真实命令结果。重点确认正确性测试只在测试函数内解析并调用 `gems_op`，性能测试显式传入 `gems_op`，两阶段输入没有改变旧 Workload，原地/out/alias 语义和断言没有丢失，且没有修改清单之外的业务文件。

4. 如果需要返工，先微调 `kernelgen/.kernelgen/agents/kernel-pytest-converter.md` 或串行流程，再记录 `needs_changes`；此时运行器只允许重试同一算子：

```bash
PYTHONPATH=.. python3 examples/pytest_converter/record_review.py --operator gt_.scalar --decision needs_changes --notes "说明需要修改的具体问题"
PYTHONPATH=.. python3 examples/pytest_converter/run_one.py --operator gt_.scalar
```

5. diff 和测试结果都通过后记录 `approved`，审阅闸门才允许进入下一个算子：

```bash
PYTHONPATH=.. python3 examples/pytest_converter/record_review.py --operator gt_.scalar --decision approved --notes "说明审阅范围和通过的测试"
```

`kernel_todo_v2/pytest_conversion_runs/` 保存每次 Agent 输出、实际改动路径和越界检查，`kernel_todo_v2/pytest_conversion_state.json` 保存当前审阅闸门和历史决定。Agent 不负责 commit、push 或自动回滚；发现越界修改时必须人工检查和恢复，再重试同一算子。
