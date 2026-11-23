# mcu（Moore Perf Compute 1.3.1）在 MTT S5000 上的实测分析

实测时间 2026-11-20 21:33–21:47，镜像 `vllm0.24.0-mthreads-musa5.2.0:2.2.0-0.3.0`，单卡 GPU7。
所有结论都能在 `raw/mcu/` 的原始报告与日志里复查。

## 1. 它是什么

`mcu` 是摩尔线程的 ncu 类工具（Moore Perf Compute，1.3.1-413aaf3c6），CLI 用法与 Nsight Compute 高度同构：

| 维度 | mcu 1.3.1 | 对应 ncu |
|---|---|---|
| 采集单元 | kernel（`-k/--kernel-name`，支持 `regex:`；`--kernel-name-base function/demangled/mangled`） | 同名参数 |
| 采集方式 | application replay（默认）/ kernel replay（`--replay-mode`） | 同 |
| 输出 | `-o xxx` → `xxx.mcu-rep`（默认 `report#.mcu-rep`） | `.ncu-rep` |
| 章节 | `SpeedOfLight`、`LaunchStats`、`MemoryWorkloadAnalysis`（默认开）、`MemoryWorkloadAnalysis_Chart/_Tables`（默认关） | 类似 section 概念 |
| 指标 | `--list-metrics` 26 个 cc3.1 指标；`--metrics a,b` | 同 |
| 采样 | `--sampling-interval`（1..65535，×256 cycles）、`--sampling-buffer-size` | 类似 |
| 列表/导出 | 无 CLI 导出（`--export-report` 未实现），报告是 SQLite | `.ncu-rep` 可用 `ncu --import --csv` |

指标清单（`raw/mcu/list_metrics.txt`）覆盖：`ddr__throughput.avg.pct_of_peak_sustained_elapsed`、
`ddr__bytes.sum.per_second`、`ddr__read/write_bytes.sum`、`gpu__compute_memory_throughput`、
`gpu__compute_memory_request_throughput`、`gpu__time_duration.max`、`d2d__*_bytes`、
`device__(ddr|llc|mp)_frequency`、`dmic__cycle_any_total.max`、`l1__atomic_requests.sum` 等。
也就是说：**MUPTI 缺的硬件计数器，mcu 有**（同一 MUPTI 1.3.1 版本，说明计数器是 mcu 侧自带
`libmupti-injection.so` 实现的），这正是本次补测的价值。

## 2. 能跑通的路径

### 2.1 torch / Triton 路径（默认参数即可）

```
mcu --sections SpeedOfLight,LaunchStats,MemoryWorkloadAnalysis \
    --kernel-name regex:softmax --launch-count 1 -o sess_softmax python t.py
```

采到 `void musa::dnn::internal::softmax_accurate_column<float, float, …>`（muDNN softmax），
8 个 replay pass；`torch.add` 采到 `musa::dnn::unary::kernel_unary<…AddOp…>`，同样 8 pass。
报告 311–324 KB，`counter_datas` 894 行。

### 2.2 自编译 MUSA 程序（必须改 kernel-name-base）

```
mcu --sections SpeedOfLight --kernel-name-base demangled --kernel-name regex:.*axpy.* \
    --launch-count 1 -o mini_demangled ./mini2 3
```

采到 `1:0:k_axpy(float*, float const*, float, int)`，3 pass，报告 221 KB / 558 行计数器。
**两个坑**（`raw/mcu/smoke_all.txt`、`smoke_mangled.txt`、`bt_sol.txt`、`sec_*.txt`）：

- 默认 `--kernel-name-base function` 下，`mcc` 编译的可执行文件一律
  `No kernels were profiled.`——即使 `--kernel-name` 写成 `regex:.*` 或 mangled 名也一样。
  只有 `demangled` 能匹配。
- 即使改成 `demangled`，**只要选 2 个以上 section 就又变成 `No kernels were profiled.`**：
  单 section（`SpeedOfLight`）能采（3 pass / 221 KB），
  `SpeedOfLight,LaunchStats` 和 3 section 全部失败（49 KB 空壳，`raw/mcu/sec_*.txt`）。
  torch 目标没有这个限制（3 section 正常 8 pass）。因此对自编译 MUSA 程序，
  目前只能拿单 section 的计数器结果。

### 2.3 权限

分别用默认、`--cap-add=SYS_PTRACE`、`+SYS_ADMIN`、`--security-opt seccomp=unconfined` 跑同一用例，
结果完全一致（都卡在内核匹配或都成功）。**不需要额外权限**，注入靠
`MUSA_INJECTION64_PATH` 环境变量，由 mcu 自己设置（`raw/mcu/smoke_*.txt` 里有 env 快照）。

## 3. 两个硬伤

### 3.1 采集结束后进程崩溃（protobuf 冲突）

```
==PROF== Capture finished. Path is ".../sess_softmax.mcu-rep"
WARNING: All log messages before absl::InitializeLog() are called are written to STDERR
F0000 ... repeated_ptr_field.h:196] Check failed: index < current_size_ (7 vs. 1)
*** Check failure stack trace: ***
Aborted (core dumped)          # 退出码 134
```

- 触发点：采集完成、准备输出 details 页面时。
- 影响：**控制台表格永远看不到**（`--page details` 也救不回来），CLI 无法给出人读的指标。
- 不影响：`.mcu-rep` 已在崩溃前写完，是完整 SQLite（见下），GUI 能打开。
- 推测原因：mcu 注入的 protobuf（`repeated_ptr_field.h`，absl 校验）与被测进程里 python/absl
  的版本冲突；C++ 目标同样崩溃，说明不完全是 python 的问题。

### 3.2 无 CLI 导出

`--export-report`、`--export-output` 在二进制里有字符串，但 CLI 不认（
`==ERROR== unrecognised option '--export-report'`）。因此"原始结果"只能是 `.mcu-rep`
（SQLite）或 GUI 导出。本次我用 SQLite 直接把表导成 CSV，放在 `raw/mcu/export/`。

## 4. 报告结构（.mcu-rep = SQLite 3）

| 表 | 行数（3 section / 1 section） | 内容 |
|---|---|---|
| `counter_datas` | 894 / 558 | `gpu_id, core_id, header, kick_info, counter_data(BLOB), original_size, pid, pass_id` —— 原始 PFM 计数器 |
| `kernel_info` | 1 | 内核全名 + `coId/submId/procId` + 元数据 BLOB |
| `session_summary` | 1 | 主机名、OS、CPU、MUSA 版本、mcu 版本（protobuf） |
| `version_info` | 2 | `pfm_submission_header_version=1`，`report_compatible_version=2026.1.19.0` |
| `device_attrs` | 4–7 | 设备属性 BLOB |
| `mupti_config` / `profile_info` / `process_info` | 1 / 1 / 4–8 | 采集配置、会话信息、进程表 |

实测样例（`raw/mcu/export/sess_softmax.counter_datas.csv`）：

```
gpu_id,core_id,header,kick_info,counter_data,original_size,pid,pass_id
7,0,1500000007000000…2c001800,000000000000000002000081…,3580a500010040c7…,…
```

`gpu_id=7` 与宿主 GPU7 的 PCI/UUID 对应，证明采的就是我们隔离的那张卡。

`session_summary` 里能读出：主机 `6e99c8854474`、`Ubuntu 22.04.5 LTS`、`x86_64`、
`INTEL(R) XEON(R) GOLD 6530`、`MUSA 5.2.0 2026-06-22 836c761ccbe5415c5e90347599889c0f5558f2df@release_musa_5.2.0`、
mcu `1.3.1-413aaf3c6`。

## 5. 与上一轮 MUPTI 实测的关系

| 能力 | MUPTI（镜像内库） | mcu 1.3.1 |
|---|---|---|
| 时间线（kernel/memcpy/API） | 有（activity API） | 不提供（面向单 kernel 计数） |
| 硬件计数器 | 无（API 骨架，`BeginSession` 返回 NOT_SUPPORTED） | **有**（PFM 注入，26 个指标） |
| 输出形态 | 自写客户端 / torch.profiler | `.mcu-rep`（SQLite）+ GUI |
| CLI 可读文本 | 需要自己写 | **不行**（崩溃） |

结论：**两者互补**。要"时间线归因"用 MUPTI/`torch.profiler`（上一轮已验收）；
要"单 kernel 计数器"用 mcu（本轮验证，报告可用但要看 GUI 或自己解 SQLite）。

## 6. 建议

1. 用宿主 GUI 打开 `.mcu-rep`（`mcu-ui`，或 `/usr/share/applications/mcu-ui-1.3.1.desktop`），
   这是目前唯一稳定的人读路径；若需要自动化，只能像本次这样解 SQLite（`counter_data` 需按
   `mupti_config` 里的 PFM 布局解码，成本较高）。
2. 对 `mcc` 编译的程序，脚本里固定加 `--kernel-name-base demangled`，否则会静默空跑。
3. 崩溃问题建议向厂商反馈：`mcu 1.3.1-413aaf3c6` + MUSA 5.2.0（2026-06-22 构建）下
   `repeated_ptr_field.h:196 Check failed`，提供 `raw/mcu/sess_softmax.txt` 即可复现。
4. 若要 CI 化，建议 `-c 1` + 单 section（4 pass，约 10 s/次），并接受退出码 134、以报告文件为判据。
