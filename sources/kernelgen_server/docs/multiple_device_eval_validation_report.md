# 多芯片 Eval 验证报告

本文只归档 KernelGen Server 的多芯片验证环境快照、兼容性问题和结果。新机器的
部署和自测步骤见
[多芯片部署与自测教程](multiple_device_deploy_tutorial.md)。

## 2026-09-05 NVIDIA H800 可取消 operation 真机验证

本轮使用 `KG v6.1.2@af9c4bb8616b5094365fa9f274fcb0a7ceed7b1a / KGS v6.2.4@ad143f2d5b5f065f532c95c600eaf42885e2f008 / Protocol v6.2` 的发布后开发快照。KGS 通过临时 Gitee 分支 `test-cancellable-operations-nvidia-20260905` 部署到 NVIDIA H800 容器的 loopback 端口，backend 为 CUDA、timing 为 Triton、8 个设备 slot、`max-workers=16`。远端解释器为 Python 3.12.3，Torch 为 `2.8.0a0+5228986c39.nv25.05`，Triton 为 3.6.0，CUDA runtime 为 12.9，driver 为 580.105.08；未安装或升级 Torch、Triton、厂商运行时或其他 Python 包。

`tests/live_server_validation.py` 以并发 16 通过 status、reference-as-solution、preflight、evaluate、portable Triton、预期 RuntimeError、隔离 worker 退出和强探针恢复；计划内的 `cuda:4` worker exit 17 形成 1 次 incident，并恢复为 8/8 healthy。`tests/live_debug_validation.py` 通过设备绑定、artifact 下载和 timeout 路径。正常 NCU profile 生成 `ncu-raw.txt`、`ncu-metrics.json` 和 `report.ncu-rep`。

Agent 在 `cuda:5` 的运行中 preflight 注册 operation 后调用 `DELETE /operations/<operation_id>`；取消请求返回 `CANCEL_REQUESTED`，原 POST 返回 `OPERATION_CANCELLED`，终态查询为 `CANCELLED`。Server 终止隔离进程树并执行强探针，最终 scheduler 为 `device_slots=healthy=available=8`、`active=waiting=checking=broken=0`、`incidents=recovered=2`，其中两次 incident 分别来自计划内 worker crash 和本次客户端取消。远端原始 Server/Debug 结果保存在 `/data/xuyao/kernelgen_e2e_20260829/kg-cli-daily-20260905/server-smoke.json` 与 `debug-smoke.json`。

## 2026-09-02 KernelGen Server v6.2.4 发布验证快照

KGS `v6.2.4` 基于 `ec05385` 发布；该功能基线把 FlagGems pytest 子进程从 `python -S` 调整为 `python -s`，保留厂商通过 system site initialization 注入运行时的能力，同时继续禁用 user site。发布同时增加根目录 `compatibility.yaml`：`v6.0` Catalog 默认使用 FlagGems adapter/flat layout，`v6.2` Catalog 默认使用 native/per-operator layout，Server 当前 `/status.api_version` 保持 `v6.2`。FlagGems 依赖固定为 `YaooXu/FlagGems` 的 `feat/new-api-for-kernelgen-server@d64794e63b502cb836bc015a92a62c42de4be05a`，运行时拒绝任何其他 HEAD。

精确 release tree 的 host 测试结果为 `283 passed, 20 skipped`，使用 detached、干净的 FlagGems `d64794e63b502cb836bc015a92a62c42de4be05a`；skip 来自当前 host 未安装 Torch 或显式关闭的设备 E2E。本轮未安装或升级 Torch、Triton、厂商运行时或其他 Python 包。该快照没有新增真机复验，部署到目标芯片后仍须执行启动探针、reference-as-solution、preflight、错误路径和资源释放检查。

## 1. 验证范围和统计口径

本报告主体记录 2026-07-30 完成的首批 v5 迁移验证。当时使用的 Catalog 子集为：

- 79 个 Definition；
- 340 个 correctness workload；
- 251 个 timing workload；
- 合计 591 个 workload。

当时的 `data/.old/flaggems-v5` Catalog 最终扩展到 327 个 Definition、1723 个
correctness workload 和 1110 个 timing workload；它现在仅作为不受支持的历史数据
归档。因此下表是可复盘的历史快照，
不能解释为当前 Catalog 的全量覆盖率；后续全量回归应新增带日期的结果章节，不覆盖
本轮数据。

每个 Definition 都把其 reference 源码同时作为 candidate 提交，即
ref-as-solution。目标不是评估优化算子的速度，而是确认以下路径在不同厂商运行时中
一致可用：

- v1 Schema、Catalog 和 Workload 加载；
- 通用及自定义输入生成；
- Python candidate/reference 加载；
- PyTree、TensorList、动态输出和自定义校验；
- preflight smoke；
- correctness 比较；
- 目标设备计时；
- 请求隔离和 JSON 响应。

所有机器显式使用 `--timing triton`，即
`triton.testing.do_bench`；快速全量测试设置 `warmup_ms=5`、
`benchmark_ms=5`。这两个参数是毫秒预算，不是固定迭代次数；单次执行超过预算时仍会
完整执行，并至少保留一次正式测量。本报告只统计可执行性，不使用短预算延迟计算性能
结论。昇腾目标 Triton 的 `do_bench` 数值已知不可靠，因此昇腾同样只统计能否完成
该路径。

## 2. 验证环境快照

下表记录 2026-07-30 验证时的环境。它用于复盘，不代表厂商镜像和空闲物理卡的永久
配置。

| 平台 | 地址或环境 | 容器/实例 | 本轮可见设备 | 逻辑后端 | 运行时设备 | Workers |
| --- | --- | --- | --- | --- | --- | ---: |
| 天数 BI-V150 | `10.31.28.29` | `codex_fib_iluvatar_20260728` | 物理卡 15 | `iluvatar` | `cuda:0` | 2 |
| 海光 BW1000 | `10.232.2.26` | `codex_fib_hygon_20260728` | 单卡 | `hygon` | `cuda:0` | 2 |
| 摩尔线程 MTT S5000 | `10.121.38.9` | `codex_fib_mthreads_20260728` | 单卡 | `musa` | `musa:0` | 2 |
| 沐曦 MetaX C550 | `192.168.2.118` | `codex_fib_metax_20260728` | 单卡 | `metax` | `cuda:0` | 2 |
| 昆仑芯 P800 | `10.21.1.131` | `codex_fib_kunlunxin_20260728` | 单卡 | `kunlunxin` | `cuda:0` | 2 |
| 华为昇腾 910B4-1 | `10.0.0.6` | `codex_fib_ascend_20260728` | **物理卡 3、4** | `npu` | `npu:0`、`npu:1` | 4 |
| 平头哥 PPU-ZW810E | PAI DSW 容器实例 | 不嵌套 Docker | 物理卡 0、1、3、4 | `thead` | `cuda:0`–`cuda:3` | 6 |

昇腾物理卡 0、1、2、5、6、7 当时已有其他任务，本轮严格未使用。平头哥 PAI DSW
本身已经是容器；不同 DSW 实例可能共享数据目录，因此连接后必须通过 `hostname` 和
实际运行时版本确认实例。

当前 v5 验证没有安装、升级或替换任何 Python 包。天数创建了
`venv --system-site-packages`，检查时所需 HTTP 包均已满足，没有实际安装或升级。
其他机器直接复用验收环境已有的轻量 HTTP 依赖；PyTorch、Triton 和厂商运行时均未
改变。

## 3. 最终结果

`Preflight` 统计全部 591 个 correctness/timing workload 的导入和 smoke 结果。
MUSA 的 `flaggems_lcm_` 在请求级超时，导致 4 个 correctness 和 3 个 timing
workload 没有返回，因此该行的分母小于 Catalog 总数。

| 平台 | Preflight | Correctness | Timing | 主要限制 |
| --- | ---: | ---: | ---: | --- |
| 天数 | 580/591 | 335/340 | 246/251 | CoreX 不支持 FP64 SVD；一个超大 `silu_backward` 用例预检 OOM。 |
| 海光 | 586/591 | 337/340 | 249/251 | 旧版 PyTorch 的 SDPA 不接受 `enable_gqa` 参数。 |
| 摩尔线程 | 550/584 | 315/336 | 231/248 | 特殊函数缺少 MUSA kernel，部分 API 能力不足，并有计时/请求超时。 |
| 昆仑芯 | 578/591 | 329/340 | 247/251 | 多个厂商 API 未实现；`renorm` 有 2 个数值不一致用例。 |
| 昇腾 | 591/591 | 340/340 | 251/251 | 全部可执行；`do_bench` 数值不作为性能结论。 |
| 沐曦 | 591/591 | 340/340 | 251/251 | 全部通过。 |
| 平头哥 | 585/591 | 337/340 | 248/251 | `torch.rnn_relu` 返回 `ACDNN_STATUS_NOT_SUPPORTED`。 |

结论：

- 79 个 Definition 均能被新 Schema 和 Catalog 加载；
- 没有发现跨芯片共现的 Server、Schema 或 JSON 协议失败；
- 沐曦和昇腾的全部 reference workload 均可执行；
- 其余失败集中在厂商 PyTorch API 能力、超大 workload 资源压力和 MUSA 超时；
- Server 没有因单个 workload 失败而退出。

## 4. 逐平台问题

### 4.1 天数

初始全量结果为 correctness 334/340、timing 245/251。复核发现：

- `flaggems_linalg_svdvals` 的 reference 按原 pytest 语义升为 FP64，CoreX
  `cusolver` 无法执行，对应 5 个 correctness 和 5 个 timing workload；
- `flaggems_rnn_relu` 的一个 FP16 correctness workload 出现非有限值，原因不是
  ref-as-solution 两侧不同，而是抽取时错误地把 RNN 权重生成为标准正态分布；
- `flaggems_rad2deg_` 的一个 timing workload 在并发全量测试中 OOM；
- `flaggems_silu_backward` 有一个形状为 `[1024, 1024, 1024]` 的 FP32
  workload，preflight 同时持有多份约 4 GiB tensor，超过 32 GiB 显存。

将 RNN 权重和 bias 改为原 `torch.nn.RNN` 的
`[-1/sqrt(hidden_size), 1/sqrt(hidden_size)]` 均匀分布后，RNN 4/4 correctness、
3/3 timing 通过。`rad2deg_` 单独复测通过，因此有效结果更新为
correctness 335/340、timing 246/251。FP64 SVD 是目标 API 能力缺口；超大
`silu_backward` 是 workload 资源问题，不属于 Schema 错误。

### 4.2 海光

`flaggems_scaled_dot_product_attention` 的 3 个 correctness 和 2 个 timing
workload 失败，因为镜像中的旧版 PyTorch 不接受 `enable_gqa` 参数。旧 FlagGems
pytest 对对应 PyTorch 版本也有跳过条件。

初始 RNN FP16 非有限值和一次 `silu_backward` 并发资源失败在修正输入分布及单独
复测后均通过。最终除上述 5 个 SDPA workload 外，其余均可执行。

### 4.3 摩尔线程

失败集中在 7 个 Definition：

- `special_chebyshev_polynomial_u`；
- `special_chebyshev_polynomial_w`；
- `special_hermite_polynomial_h`；
- `special_shifted_chebyshev_polynomial_w`；
- `embedding_dense_backward`；
- `igammac_`；
- `lcm_`。

四个特殊多项式算子缺少对应 MUSA kernel。
`embedding_dense_backward(scale_grad_by_freq=True)` 不受运行时支持。
`igammac_` correctness 可执行，但部分 `do_bench` timing 启动或同步超时。
`lcm_` 整个请求超过 900 秒，7 个 workload 没有返回，仍需进一步区分厂商 kernel
挂起与 workload/原地计时语义。

旧 v4 中的 FP16 `bucketize` 失败已消失：v5 正确保留 candidate 输入 dtype，并将
`boundaries` 固定为 FP32，reference 只对 `x` 做 FP32 转换。

### 4.4 昆仑芯

10 个 Definition 存在失败：

- `upsample_linear1d`、`upsample_linear1d_backward`；
- `adaptive_max_pool3d_backward`；
- `max_pool3d_with_indices`；
- `log10_`；
- `mish`、`mish_`；
- `softplus_backward`；
- `embedding_dense_backward`；
- `renorm`。

除 `renorm` 外，失败均为厂商 PyTorch/XMLIR 明确报告 `NOT IMPLEMENTED` 或不支持
`scale_grad_by_freq=True`。`renorm` 有 2 个真实数值不一致 workload，需要在后续
算子适配中单独处理。

### 4.5 昇腾

物理卡 3、4 上全部通过：591/591 preflight、340/340 correctness、251/251
timing。该结果说明新 Schema、reference、隔离子进程及 `do_bench` 调用路径能够
完成，不代表 `do_bench` 延迟值可信。

正式性能评测仍应使用 `--timing auto` 或 `--timing profiler`。Profiler 采集失败
必须返回错误，不允许使用 wall time 或无目标 NPU kernel 的记录代替。

### 4.6 沐曦

79/79 个 Definition 全部通过，591 个 preflight、340 个 correctness 和 251 个
timing workload 均无失败或请求级错误。全量任务退出码为 0。

### 4.7 平头哥

唯一失败的 Definition 是 `flaggems_rnn_relu`。1 个 correctness workload 可执行，
其余 3 个 correctness 和 3 个 timing workload 由 ACDNN 返回
`ACDNN_STATUS_NOT_SUPPORTED`。这是厂商 API capability，不是 Server 或 Schema
问题。

平头哥还通过了可复用功能测试：reference、自定义输入、自定义校验、PyTree、
可移植 Triton 编译、严格 `do_bench`、多设备并发、`RUNTIME_ERROR`、worker
硬退出隔离和失败后恢复。

## 5. 从失败案例得到的通用结论

### 5.1 抽取必须复刻 pytest，而不是只复制 API

RNN 是典型案例。原 pytest 使用 `torch.nn.RNN` 创建权重，其分布范围取决于
`hidden_size`；使用通用 `randn` 会在长序列 FP16 recurrence 中溢出，即使
reference 与 candidate 是同一份代码也会同时得到 `Inf`。

抽取器必须分别追踪：

- correctness pytest 的 setup、输入构造、reference 预处理和断言；
- benchmark `input_fn` 的输入分布和计时对象；
- dtype、有效域、view、alias、mutation、kwargs 和动态 PyTree 输出；
- 只属于 reference 的 CPU 转移、FP32 upcast、cast-back 和局部输出比较。

### 5.2 非有限误差必须保持 JSON 合法

早期 RNN 非有限值使比较器产生 `float("inf")`，Pydantic 可以持有该值，但严格 JSON
响应不能序列化 `Infinity`，最终表现为 HTTP 500。当前协议在误差无法表示为有限数
时返回 `null`，并保留诊断消息；单个请求异常也不会阻止全量脚本保存其他结果。

### 5.3 输入克隆必须保留特殊 tensor 语义

普通 `clone()` 会物化复数 tensor 的 lazy negative view，使逻辑值不变但
`is_neg()` 变为 `False`。这会让 `resolve_neg` 的 candidate 路径退化为无操作。
当前 PyTree clone 在保持独立存储的同时恢复 negative bit，并保留重复对象的 alias。

### 5.4 资源失败与 API 缺失要分开

- `NOT IMPLEMENTED`、不支持参数和缺少设备 kernel 属于厂商 API capability；
- 超大 tensor 在多 Definition 并发下 OOM、单独复测通过，属于资源压力；
- 输入 helper 的 forward 已不支持时，应归类为输入生成阶段能力缺口；
- 请求超时需要保留请求级状态，不能伪装成某个 correctness 或 timing 失败。

## 6. 与旧 FlagGems v4 结果对比

v5 从旧 trace 模型迁移到扁平 Definition/Workload/Implementation 协议，并增加
`einsum`、`unbind_copy` 等覆盖，因此 workload 数从旧轮的 436 增加到 591。两轮
数量和计时策略并不完全一致，只比较稳定的失败类型。

| 平台 | 旧 v4 有效结果 | v5 观察 |
| --- | --- | --- |
| 天数 | 436/436 | 新增的严格 FP64 SVD 语义暴露 CoreX 能力缺口；RNN 抽取问题已修复。 |
| 海光 | 431 通过、5 跳过 | 同样只有旧 PyTorch `enable_gqa` 的 5 个 SDPA workload。 |
| 摩尔线程 | 422 通过、14 跳过 | `bucketize` dtype 已修复；扩展特殊函数覆盖后暴露更多 MUSA kernel 和超时问题。 |
| 沐曦 | 436/436 | v5 仍全部通过。 |
| 昆仑芯 | 421 通过、11 运行错误、2 数值错误、2 跳过 | 失败仍集中在厂商 API 和 `renorm`，类型基本一致。 |
| 昇腾 | correctness 246/246；profiler timing 169 通过、21 跳过 | v5 的 `do_bench` 路径全部可执行，但不能与旧 profiler 结果做性能比较。 |
| 平头哥 | 430 通过、6 跳过 | 仍是 `rnn_relu` 的 6 个 ACDNN 能力缺口。 |

## 7. 2026-07-30 Debug Job 与环境上报验证

将 `origin/dev-jiabei-profile` 的目标机调试能力按当前架构重写后，在同一份最终源码上
复用了 `tests/live_debug_validation.py` 和 `tests/live_server_validation.py`。
每台机器均验证：

- `/status` 返回 backend、逻辑设备、芯片名称和 Triton/运行时版本；
- Debug Job 在 slot 分配的设备上实际创建一个 Tensor，并把物理可见设备收敛为一张；
- bounded wait、artifact 下载及大小/SHA-256 校验、运行中取消均正常；
- 取消后 Server 仍可用；
- `flaggems_zero_` 的一个 correctness 和一个 timing workload 以
  reference-as-solution 完成 preflight/evaluate。

| 平台 | 本轮设备 | Debug Job | reference-as-solution | 计时 |
| --- | --- | --- | --- | --- |
| 天数 BI-V150 | 物理卡 15 | 通过 | 2/2 通过 | `do_bench` |
| 海光 BW1000 | 物理卡 7 | 通过 | 2/2 通过 | `do_bench` |
| 摩尔线程 MTT S5000 | 物理卡 7 | 通过 | 2/2 通过 | `do_bench` |
| 沐曦 MetaX C550 | 物理卡 7 | 通过 | 2/2 通过 | `do_bench` |
| 昆仑芯 P800 | 物理卡 7 | 通过 | 2/2 通过 | `do_bench` |
| 华为昇腾 910B4-1 | **物理卡 3、4**，任务落在 3 | 通过 | 2/2 通过 | 严格 profiler |
| 平头哥 PPU-ZW810E | 物理卡 0 | 通过 | 2/2 通过 | `do_bench` |

本轮发现并修复两个可移植性问题：

- 天数 Triton 可正常导入但没有标准 distribution metadata，版本探测增加
  `triton.__version__` 的一次性子进程兜底；
- 沐曦 Server 使用 `/opt/conda/bin/python`，裸 `python3` 会落到没有 Torch 的
  `/usr/bin/python3`。Debug Job 现在用显式 `"{python}"` 占位符选择 Server
  解释器，并注入 `KGS_PYTHON`。

昇腾 CANN 在成功任务退出时可能向 stdout 追加诊断文字，因此在线脚本只把首行
`debug-live-ok` 作为协议标记，完整 stdout 仍保留在 Job 结果中。本轮没有安装或升级
任何包；测试结束后只停止新增的 `18206` 临时服务，原有服务未改动。

## 8. 后续验证要求

在新镜像、驱动、PyTorch 或 Triton 版本上重新验证时：

1. 不覆盖本报告结果，新增带日期的环境快照；
2. 同时保存完整 JSON 和失败 traceback；
3. 对并发 OOM、随机波动和超时进行同一 Server 实例上的定向复测；
4. 只有复测通过才能更新有效结果；
5. 记录所有新增包，尤其确认 PyTorch、Triton 和厂商扩展没有变化；
6. 昇腾正式性能计时必须使用严格 profiler，禁止 fallback。

## 9. 2026-08-02 同步 Debug Job 验证

在 Gitee 临时分支 `test-sync-debug-submit-20260802` 验证
`POST /debug/jobs` 与 evaluate 一样保持单个 HTTP 请求直到终态。验证通过后，
commit `725e7e3` 已 fast-forward 合并到 Gitee `main`。

环境与范围：

- 容器：`codex_fib_ascend_20260728`；
- 物理卡：只使用 3，Server 内映射为 `npu:0`；
- backend/timing/slot：`npu` / 严格 `profiler` / 1；
- 临时 loopback 端口：`18316`；
- Torch `2.9.0+cpu`、torch_npu `2.9.0.post2`、Triton-Ascend `3.5.1`、
  CANN `9.0.0`、driver `25.2.0`。

结果：

- 单次 `submit_debug_job` 直接返回 `SUCCEEDED`，目标端真实创建 NPU Tensor；
- `ASCEND_RT_VISIBLE_DEVICES` 在 Debug 子进程中为物理卡 `3`；
- stdout、artifact 下载、大小/SHA-256 校验和本地 workspace 记录均通过；
- 1 秒 Debug Job 由 Server 直接返回 `TIMEOUT`，不需要客户端 GET 轮询；
- 本地 KernelGen 通过 SSH stdio HTTP proxy 调用远端 loopback Server，通过同一
  同步 adapter 验证；
- Server 定向测试 13/13 通过；
- 最小 reference-as-solution、preflight、严格 profiler timing、Triton candidate、
  两请求并发、预期 `RUNTIME_ERROR`、硬崩溃隔离和崩溃后恢复全部通过。

为运行 FastAPI 路由单测新增轻量包 `httpx==0.28.1` 和
`httpcore==1.0.9`；Torch、Triton、torch_npu 和厂商运行时未安装、升级或替换。
原始结果保存在测试目录的 `live-debug-sync.json` 和
`live-server-minimal.json`。测试结束后本地 proxy 和临时 Server 均已关闭，端口
`19616`、`18316` 已释放。

## 10. 2026-08-03 强探针与全量 Catalog 回归

在 Gitee 临时分支 `test-probe-only-broken-20260802` 的 commit `669d842` 上，
对当前 327 个 Definition、1723 个 correctness workload 和 1110 个 timing
workload 再次执行 ref-as-solution 全量回归。所有 Server 均显式使用
`--timing triton`，快速测试预算为 `warmup_ms=5`、`benchmark_ms=5`，请求预算为
900 秒；昇腾结果仍只表示 `do_bench` 路径可执行，不能作为性能数据。

本轮还验证了新的故障处理语义：所有可见设备并行执行强启动探针；请求超时或隔离
worker 异常退出后，只在原 slot 执行一次强探针，不跨 slot 重试。强探针通过则恢复
slot，失败才标记为 `broken`。

| 平台 | 物理卡数 | 启动健康 slot | Definition 通过 | 未通过 | 请求 incident / 恢复 | 最终 broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 天数 BI-V150 | 2 | 2/2 | 325 | 2 | 0/0 | 0 |
| 海光 BW1000 | 7 | 7/7 | 314 | 13 | 1/1 | 0 |
| 摩尔线程 MTT S5000 | 5 | 5/5 | 300 | 27 | 0/0 | 0 |
| 沐曦 MetaX C550 | 8 | 8/8 | 323 | 4 | 0/0 | 0 |
| 昆仑芯 P800 | 8 | 5/8 | 281 | 46 | 0/0 | 3 |
| 华为昇腾 910B4-1 | 5 | 5/5 | 310 | 17 | 0/0 | 0 |
| 平头哥 PPU-ZW810E | 14 | 14/14 | 322 | 5 | 3/3 | 0 |

“Definition 通过”要求该 Definition 的 correctness 和 timing 两个阶段均通过。
未通过不等于 Server 失败：主要原因仍是厂商 API/数据类型不支持、数值差异或隔离
worker 退出。代表性结果如下：

- 天数仅 `kthvalue` 数值不一致和 FP64 `linalg_ldl_factor` 不受支持；
- 海光 13 个失败包括旧 PyTorch API 缺口、Half/BFloat16 kernel 缺失，以及
  `linalg_eigvals` 的一次 900 秒超时；超时后强探针通过并恢复原 slot；
- MUSA 的 27 个失败主要为特殊函数、Flash Attention、累积算子等厂商 kernel
  缺失或 MuDNN 错误，没有触发设备健康 incident；
- 沐曦 4 个失败为 `special_i1` Half kernel、Bessel 设备断言、未启用 Flash
  Attention 和 cuDNN 版本能力；
- 昆仑芯启动时稳定识别出 3 个无法通过强探针的 slot，剩余 5 个 slot 完成全量
  测试且没有新增 incident；46 个失败主要为 `NOT IMPLEMENTED`、参数能力和数值
  差异，另有一个 `do_bench` 负延迟；
- 昇腾 17 个失败主要为算子或 dtype capability 和少量数值差异，5 个 slot 全程
  健康；
- 平头哥 3 个线性代数 preflight worker 以 exit code 1 退出，强探针均通过并恢复
  原 slot，Server 保留原始 HTTP 500；另两个失败来自 ACDNN/cuDNN API。

并行强启动探针将 14 卡环境的启动时间从逐卡累加降到约一个最慢探针周期；七台
Server 均在 40 秒内进入可服务状态。全量测试期间没有新增 `broken` slot，说明强
探针与“不跨 slot 重试”的简化容错流程符合预期。

海光全量测试中的 `flaggems_linalg_eigvals` timing 超时随后做了单 Definition
串行对照：`parallel-definitions=1`，Server 始终只有 `active=1`、`waiting=0`，
其余 6 张卡空闲，timeout 仍为 900 秒。correctness 在 4.25 秒内通过；timing
仍返回 `TIMEOUT`，evaluate 端到端耗时 914.75 秒。故该超时不是全量并发造成，
而是该 Definition 的 timing/`do_bench` 路径本身过慢。超时后的强探针再次通过，
原 slot 恢复，最终仍为 7/7 健康。对照结果保存在
`hygon/timeout_retest_serial/result.json`。

各机完整结果位于：

```text
<DEPLOY_BASE>/runs/kernelgen_server_strong_probe_669d842/<device>/
  server.log
  flaggems-v5-full.log
  flaggems-v5-full.json
```

本轮各机从 Gitee 直接拉取 test 分支，并执行 `pip install -e . --no-deps`，没有
安装或升级 Torch、Triton、厂商运行时或其他依赖。昇腾离线源缺少构建隔离环境所需
的 `wheel`，因此增加 `--no-build-isolation` 完成同一 editable 安装。平头哥环境
另有旧工作区提供同名包；全量脚本从新 clone 根目录通过 `runpy` 启动，确保实际
导入 commit `669d842`，没有用 `PYTHONPATH` 覆盖环境。

## 11. 2026-08-25 v6.2 native 与 FlagGems adapter 对照

在沐曦 MetaX C550 单卡上验证 API v6.2 的 per-operator native catalog，并用同一份
`addmm_` candidate 对照直接 FlagGems adapter。Server commit 为 `2595967`，
FlagGems commit 为 `fc2a91c`；Torch `2.8.0+metax3.7.2.0`、Triton `3.6.0`。
candidate SHA-256 为
`fb7cf5795ca533a7c0f82ea1ab5f95a9145630ada62db65871ff45775614959f`。

两种 evaluator 的 15 个 core timing case 的 case id、顺序、dtype、M/N/K 和 layout
逐项一致。设置均为 `warmup_ms=1000`、`benchmark_ms=100`、一次 trial，并交错执行
三轮。为了与 adapter 不对 timing case 做数值门的行为隔离比较计时，性能对照使用
宽松容差；严格数值结果另行保留。

| evaluator | 三轮 geo mean | 中位数 |
| --- | --- | ---: |
| FlagGems adapter | 0.594545、0.594525、0.597036 | 0.594545 |
| v6.2 native | 0.595047、0.596244、0.599263 | 0.596244 |

native 中位数比 adapter 高 0.286%，属于本轮重复测量波动。adapter 的一次严格运行为
0.598655，与历史结果 0.599131 相差 0.08%。首次 native 抽取曾得到 0.470511 的
中位数级结果；逐 case 排查发现 candidate latency 一致，差异全部集中在 float32
Torch baseline。原因是 FlagGems `benchmark/base.py` 会关闭 TF32，而最初抽取的
`timing_run()` 没有携带该 reference 策略。将策略限定在单次 oracle 调用内并在返回
后恢复，避免 import 副作用后，两种 evaluator 的结果对齐。该问题说明 workload
相同还不够，抽取器也必须复现 source benchmark 的 reference 执行策略。

严格 native evaluate 为 46/51 `PARTIAL_PASS`，5 个 float32 timing case 均被
timing correctness/effects gate 判为数值不一致，因此没有合法 headline geo mean。
candidate 对这些方形 workload 显式走 TF32 fast path；直接 adapter 只在独立 accuracy
case 做正确性检查，不校验 timing case，所以仍报告 51/51 `PASSED`。宽松容差结果只
用于 evaluator 计时对照，不能覆盖严格数值结论。

上述 `PARTIAL_PASS` 记录的是当时 v6.2 native 尚有 timing correctness gate 的历史
行为。后续协议口径已与 framework adapter 统一：correctness 只由 correctness workload
负责，timing workload 不再比较数值、mutation 或 alias。因此同一 candidate 在当前
v6.2 应报告 51/51 `PASSED`；这不表示 5 个 float32 timing 输出获得正确性背书。

修正后的 reference-as-solution 为 51/51 `PASSED`，preflight 通过；此前还验证了
预期 `RUNTIME_ERROR`、两请求并发串行占用单 slot 和资源释放。最终 scheduler 为
`device_slots=healthy=available=1`，`active=waiting=checking=broken=0`，incident
为 0。本轮没有安装、升级或替换任何包。

## 12. 2026-08-28 v6.2 多厂商 Profiling 验证

本轮验证功能代码 commit `f87ad485caf480587076f397088a790fed78cb9e`。华为、天数
和海光使用 checkout `8a349658605510a18c347ecc81a08d8c5ad54ac5`，其余平台最初
使用 `f961933283ab37bc31e3165c4df2afd10c3ddb58`，昆仑芯成功复测使用
`069760c98108131fd61b90b7c4178ae165925827`；这些后续 commit 相比前者只修改本文档，
profiling 代码完全一致。所有成功请求都用同一份 portable Triton
`kernelgenbench_square` candidate 精确重放 `aten__square-00000-timing`。Server
上报 API v6.2、软件版本 v6.2.0，临时实例只暴露一张物理卡和一个 slot。常规
`metrics` 请求为 warmup 2 次、测量 3 次，`instruction` 为 warmup 1 次、测量 1 次；
厂商 profiler 自身要求不同采样次数时在结果中明确记录。

| 平台 | 物理卡 | Profiler | metrics | instruction |
| --- | ---: | --- | --- | --- |
| 华为 Ascend 910B4-1 | 4 | `msprof` | 3 次 `_square_kernel`，平均 2.820 us | 154 条指令，113 条映射到源码 |
| 天数 BI-V150 | 9 | `ixsys+ixkn` | 3 次 `_square_kernel`，平均 4.820333 us | 35 条指令，35 条映射到源码，并导出编译 IR |
| 海光 BW1000 | 7 | `hipprof/rocprof` | 3 次 `_square_kernel`，平均 3.200 us | 33 条指令，33 条映射到源码 |
| 英伟达 H800 | 单卡 | `ncu` | NCU 将 3 次规范化为 1 次，2.720 us | 34 条指令，34 条源码映射 |
| 沐曦 MetaX C550 | 单卡 | `mcTracer` | 3 次，共 10.752 us，平均 3.584 us | 1 个 device object、19 个 compiler operation，7 个源码映射；无原生机器 ISA |
| 平头哥 PPU-ZW810E | 7 | `acu` | ACU `launch-count=1`，1 次，1.45647 us | 56 条指令，35 条源码映射 |
| 摩尔线程 MTT S5000 | 5 | `mcu` | 5 次，平均 6.744 us | MCU 当前不提供可验证的 instruction 证据，API 只公布 `metrics` |
| 昆仑芯 P800 | 5 | `xprofiler` | 3 次，平均 1.778 us，trace 完整 | XProfiler 当前只公布 `metrics` |

这些 profiler 时间只用于诊断，不是 Eval 权威 latency。天数 `ixsys` 没有在 SQLite
中保留 `kernelgen_profile` NVTX range，但 `cudaProfilerStart/Stop` 采集边界生效，
正式测量的 kernel 次数与请求完全一致，因此结果保留该 warning 后判为完成。

八个平台的可用 level 均完成，vendor report、结构化结果和 Native completion
marker 齐全。英伟达 NCU 会内部 replay kernel，因此按既定规则把请求迭代数规范化
为 1。平头哥请求仍记录 `iterations=3`，但 ACU 默认 `--launch-count=1`，实际报告
只有一行 kernel；当前 `measured_iterations` 字段表示请求参数而非报告行数，不能把它
解释为实际采样数量。摩尔线程 MCU 报告包含 8 个 application replay pass 和 5 次
kernel 调用；它不会区分 warmup 与正式测量，结果同样只作为诊断证据。

天数首次测试使用符号链接形式的隔离 venv。`ixsys` 启动应用时把该 Python 路径解析
为系统解释器，子进程因而误加载基础 KGS v5.1，并被 completion marker 检查正确判为
失败。改用 `python3 -m venv --copies` 后，同一请求通过。该问题属于测试部署的
解释器路径，不需要修改 profiler；失败 attempt 和成功复测均已保留。

沐曦当前 SDK 只能导出 compiler operation，不能伪装为原生机器 ISA，因此
instruction 结果带明确 warning。平头哥采集前临时暂停 `amperfd`，结束后已经恢复为
Collecting。昆仑芯首次使用物理卡 4 时，当前 KGS 在启动强探针阶段收到厂商运行时
`cuiInitCheckEx status=201`；物理卡 6 的后续探针也在 120 秒超时，两次都没有绕过
探针提交 `/profile`。2026-08-29 改用物理卡 5 后，v6.2 强探针通过，XProfiler 对
同一 candidate 精确采集 3 次 `_square_kernel`，总计 5.333 us，平均 1.778 us，
调用次数、measurement window、Native completion marker 和 trace 完整性均通过严格
校验。

昆仑芯首次正式 profile 因测试 checkout 没有安装到 Python 环境而失败：XProfiler
把 cwd 切到 artifact 目录后，内层 `python -m kernelgen_server.profiling.runner` 无法
导入当前 KGS。确认 XProfiler 能正常启动普通子进程后，在临时 Server 上显式设置该
checkout 的 `PYTHONPATH`，同一请求通过；这属于测试部署修正，没有修改 KGS 或厂商
runtime。既有 19403 Server 覆盖物理卡 5–7，但复测前后始终
`active=waiting=checking=0`，没有发生同卡并发任务；其原有
`healthy=2, broken=1, incidents=71` 状态也没有变化。当前主机清单没有寒武纪连接
信息，因此本轮无法验证寒武纪；燧原按本轮范围明确排除。

成功平台结束时临时 slot 均为 `healthy=available=1`，且
`active=waiting=checking=broken=0`。摩尔线程测试前后四个既有 Server 均健康空闲；
测试中既有 19303 Server 曾在物理卡 1–4 上执行任务，与本次物理卡 5 不重叠。所有
临时 Server 均已停止，临时 checkout 已删除，持久化宿主机部署 key 与测试 artifacts
保留。完整请求、结果、Server 日志和 vendor artifacts 位于：

```text
# 华为
/home/secure/xuyao/kernelgen_e2e_ascend_20260821/runs/
  multiple-profiling-smoke-20260828/ascend/

# 天数、海光
/workspace/kernelgen_e2e_20260728/runs/
  multiple-profiling-smoke-20260828/{tianshu,hygon}/

# 英伟达、沐曦、摩尔线程
/data/xuyao/kernelgen_e2e_20260728/runs/
  multiple-profiling-smoke-20260828/{nvidia,metax,mthreads}/

# 平头哥（宿主机持久化路径）
/mnt/workspace/codex_fib_thead_20260818/kernelgen_e2e_20260818/runs/
  multiple-profiling-smoke-20260828/ppu/

# 昆仑芯首次启动探针失败证据
/home/secure/xuyao/kernelgen_e2e_20260728/runs/
  multiple-profiling-smoke-20260828/kunlun/

# 昆仑芯物理卡 6 超时、物理卡 5 成功及部署诊断
/workspace/kernelgen_e2e_20260728/runs/
  multiple-profiling-smoke-20260829/kunlun-card{6,5,5-xprofiler,5-xprofiler-pythonpath}/
```

本轮没有安装、升级或替换 PyTorch、Triton 和厂商运行时。华为和天数使用独立 venv
安装当前 KGS；海光首次创建 venv 时遇到 Debian 脚本目录异常，editable 安装短暂
落到基础环境，发现后立即恢复为原 `/workspace/kernelgen_server` 的 KGS v5.1，
并改用已验证 `sys.prefix` 的隔离 venv 完成测试。海光既有三个 Server 在恢复后均为
`ok`、空闲且 `broken=0`。
