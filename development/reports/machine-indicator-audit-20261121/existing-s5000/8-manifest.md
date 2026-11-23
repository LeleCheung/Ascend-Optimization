# MTHREADS-S5000-MCU-20261120 — 摩尔线程 mcu（Moore Perf Compute）实测

## 1. 实验标识

| 项 | 值 |
|---|---|
| 实验编号 | `MTHREADS-S5000-MCU-20261120` |
| 日期 | 2026-11-20（Asia/Shanghai 21:33–21:47，容器内 UTC 13:33–13:47） |
| 目标机 | `baai-S5000` = `worker33015`（与 `MTHREADS-S5000-PROFILER-FT-20261120` 同一台机器） |
| mcu 版本 | **1.3.1-413aaf3c6**（deb 包 `mcu` amd64，`Moore Perf Compute`） |
| mcu 安装位置 | `/opt/moore-threads/moore-perf-compute/1.3.1/`，`/usr/bin/mcu` → `…/target-linux-x86_64/mcu.sh` |
| 注入方式 | 环境变量 `MUSA_INJECTION64_PATH=…/libmupti-injection.so` + `MCU_HOME_DIR`（由 mcu 自动设置） |
| 容器镜像 | `harbor.baai.ac.cn/flagos-app/vllm0.24.0-mthreads-musa5.2.0:2.2.0-0.3.0`（同一镜像，只读挂载 `/opt/moore-threads`） |
| 容器 OS / MUSA | Ubuntu 22.04.5，MUSA 5.2.0（`2026-06-22 836c761c…@release_musa_5.2.0`），MUPTI 1.3.1 |
| 被测设备 | GPU7，PCIe `0000:da:00.0`，UUID `053b311f-c3fd-feb3-a232-5b57d7f8c5b0`（mcu 报告里 `counter_datas.gpu_id=7`） |
| 宿主 | 2× Xeon Gold 6530 / 128 线程 / 2015 GiB，8× MTT S5000 80 GiB，驱动 3.3.5-server |
| 证据状态 | 已验证（`raw/mcu/*.mcu-rep` 为 mcu 原生报告，`raw/mcu/export/` 为解包后的表） |

## 2. 不干扰他人的做法

1. 开工前（21:32）与结束后（21:47）均为：8 卡 `0% / 0 MiB`，`mthreads-gmi` 进程表
   `No running processes found`，只有原有的 3 个他人容器在跑且无 GPU 进程。
2. 容器只映射 **GPU7 单卡**：`--device=/dev/dri --device=/dev/mtgpu.7`，
   `/opt/moore-threads` 只读挂载（不往宿主写任何东西）。
3. 未修改 mcu 配置、未改驱动参数、未 `docker stop/kill` 他人容器；所有容器 `--rm`。
4. 额外验证了权限假设：`--cap-add=SYS_PTRACE`、`+SYS_ADMIN`、`--security-opt seccomp=unconfined`
   三种组合结果与默认完全一致（都失败在同一个"无 kernel"现象上），因此 **不需要提权**，
   后续所有正式运行都用默认权限。
5. 每个 mcu 运行都在容器内限 `--cpus=16 --memory=64g`，报告写在挂载目录，跑完容器即销毁。

## 3. 实测结论（一句话）

**mcu 能跑通、能出原生报告，但当前组合（mcu 1.3.1 + 镜像 MUSA 5.2.0）有两处硬伤：**

1. **内核名过滤默认值不匹配 mcc 产物**：默认 `--kernel-name-base function` 对 `mcc` 编译出的
   MUSA 可执行文件一律报 `No kernels were profiled.`；必须显式
   `--kernel-name-base demangled --kernel-name regex:.*`（或具体内核名）才能采到。
   用 torch/Triton 路径（名字本身是 demangled C++ 模板名）则默认即可。
2. **采集完成后进程崩溃**：`repeated_ptr_field.h:196 Check failed: index < current_size_ (7 vs. 1)`
   （mcu 注入的 protobuf 与镜像内 python/absl 冲突），退出码 134，
   **控制台 details 页面因此打不出来**；但 `.mcu-rep` 在崩溃前已落盘且是完整 SQLite，
   可用宿主上的 `mcu-ui` 打开。

除此之外采集本身是成功的：3 个 section 时 8 个 application-replay pass、单 section 时 4 个 pass，
报告 `counter_datas` 有 894 / 558 行原始计数器数据，`gpu_id=7` 与目标卡一致。

## 4. 测试矩阵

| 用例 | 命令要点 | 结果 |
|---|---|---|
| CLI 面 | `mcu --version/--help/--list-metrics/--list-sections` | 1.3.1；5 个 section（3 默认开）；26 个 mp_31(cc3.1) 指标 |
| 空跑（无过滤） | `mcu --launch-count 1 ./mcu_bench` | `No kernels were profiled.`，报告 49 KB（空壳） |
| mangled 过滤 | `--kernel-name-base mangled -k regex:_Z6k_copy` | 仍无 kernel |
| **demangled 过滤（单 section）** | `--sections SpeedOfLight --kernel-name-base demangled -k regex:.*axpy.* ./mini2` | **成功**：3 pass，报告 221 KB / 558 行计数器 |
| demangled + 多 section | 同上但 `--sections SpeedOfLight,LaunchStats[,MemoryWorkloadAnalysis]` | 失败：`No kernels were profiled.`，49 KB 空壳（`sec_*.txt`） |
| torch 内核（默认过滤） | `-k regex:softmax` → `softmax_accurate_column<…>` | **成功**：4 pass（1 section）/ 8 pass（3 section），报告 242–324 KB |
| torch 内核（add） | `-k regex:kernel_unary` → muDNN `kernel_unary<AddOp>` | **成功**：8 pass，报告 311 KB |
| 权限矩阵 | `+SYS_PTRACE` / `+SYS_ADMIN` / `seccomp=unconfined` | 与默认一致，无改善（也无需提权） |
| 报告解包 | `.mcu-rep` = SQLite3（`SQLite format 3`） | 8 张表，已导出 CSV/文本到 `raw/mcu/export/` |
| 导出选项 | `--export-report/--export-output` | CLI 未实现（`unrecognised option`），只能拿 `.mcu-rep` |

## 5. 产物结构

```
MTHREADS-S5000-MCU-20261120/
├── manifest.md                      本文件
├── analysis.md                      mcu 能力/限制、报告数据说明、复现命令
├── MTHREADS-S5000-MCU-20261120.tar.gz
└── raw/
    ├── mcu/
    │   ├── sess_softmax.mcu-rep      torch softmax，3 section，8 pass（323 KB）
    │   ├── sess_add.mcu-rep          torch add(kernel_unary)，3 section（311 KB）
    │   ├── torch_softmax.mcu-rep     torch softmax，1 section（242 KB）
    │   ├── mini_demangled.mcu-rep    C++ k_axpy，1 section，demangled 过滤（221 KB）
    │   ├── diag_torch.mcu-rep        最早一次 torch 取证（315 KB）
    │   ├── list_metrics.txt          指标清单（26 个 cc3.1 指标 + mp_22 变体）
    │   ├── list_sections.txt         5 个 section
    │   ├── *.txt                     各次运行的完整 stdout/stderr（含 crash 栈）
    │   └── export/                   SQLite 逐表导出：*.csv（含 counter_data 十六进制）+ 关键表文本
    ├── bench/                        mini.mu / mini2.mu / mcu_bench.mu（mcu 目标负载源码）
    ├── scripts/                      case_mcu_*.sh 与容器启动脚本（含每步命令）
    └── host_mcu_facts.txt            mcu 版本、安装树、GPU7 状态、进程/容器快照
```

## 6. 复现

```bash
# 宿主：同一镜像 + 只读挂载 mcu + 单卡 GPU7
docker run --rm --device=/dev/dri --device=/dev/mtgpu.7 \
  -v /opt/moore-threads:/opt/moore-threads:ro \
  -v /data/ps-s5000-prof-20261120:/work -w /work --cpus=16 --memory=64g \
  --entrypoint /bin/bash \
  harbor.baai.ac.cn/flagos-app/vllm0.24.0-mthreads-musa5.2.0:2.2.0-0.3.0 \
  -lc 'MCU=/opt/moore-threads/moore-perf-compute/1.3.1/target-linux-x86_64/mcu.sh; \
       cd /work/raw/bench && \
       $MCU --sections SpeedOfLight,LaunchStats,MemoryWorkloadAnalysis \
            --kernel-name-base demangled --kernel-name regex:.*axpy.* \
            --launch-count 1 -f -o /work/mcu_out ./mini2 3'

# 打开报告（宿主，需图形界面）
mcu-ui        # 或 /usr/share/applications/mcu-ui-1.3.1.desktop
```

## 7. 校验和

`MTHREADS-S5000-MCU-20261120.tar.gz`
`sha256 = f1c318a95583e206023637eb568328c447ac81d3d9da731e513d7057e4cb8df8`
