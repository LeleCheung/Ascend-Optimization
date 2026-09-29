# 候选准入与 metadata 白名单

## 当前实现：两个精确豁免

维护者维护的列表位于 `kernelgen_server/evaluation/metadata_policy.py`。当前只有 `lift_fresh` 和 `unsqueeze_`。候选不能通过自己的名字或 policy 字段获得豁免；native 使用评测 Definition 名称，FlagGems adapter 使用已绑定 Catalog operator 名称，二者均调用同一个检测器。

白名单仅豁免缺少 `@triton.jit`／kernel launch 的检测项，不豁免 ABI、数值、alias、输入效果或 workload 校验。为避免仅凭算子名放行任意 Python，首版只接受下列最小实现：

```python
# lift_fresh：保留输入对象，不复制、不写入。
def run(x):
    return x

# unsqueeze_：保留输入对象与 storage，按传入 dim 原地增加维度。
def run(x, dim):
    return x.unsqueeze_(dim)
```

第二种也允许 `x.unsqueeze_(dim)` 后 `return x`。参数名可变，允许简单类型标注、docstring 和普通 Torch 导入；装饰器、额外 helper／source、全局写入、annotation 调用或额外计算不获得该豁免。未匹配的写法继续遵循既有检测，不自动标成编译器问题。以上是可审计的最小源码形式，不是任意 Python 安全沙箱。

`resize_output_` 需要补查设备、容量和 alias 契约，暂不加入；`unbind_copy` 涉及 copy，也不加入。后续扩展必须同时提交语义依据和正反用例，不能根据名称包含 view／resize 自动放宽。

## 统一准入已实现

`candidate_admission.py` 仅由 preflight 调用，在候选 import 前完成准入。evaluate/profile 不再重复执行准入或 hack 检测，evaluate 也不再重复扫描灰名单。明确的 Torch namespace 计算回退、缺少有效 JIT/launch、仅零 grid 的已识别 Triton launch、保护命名空间改写与注册／全局设置调用会被拒绝。普通 Tensor 同名方法等无法静态确认接收者的命中以 `ADMISSION_REVIEW` 日志保留，不作为已确认 hack。现有 `detect_obvious_hack` 保留兼容诊断接口，新的门禁消费其结构化分类，不用字符串拆分重新判定。

Gems 专属注册／源码／测试约束仅由 Gems adapter 启用。共享门禁的 evaluator_kind 由实际 adapter 类型传入，不能由候选控制，也不根据算子名或 framework 名推断。native 不启用 Gems 专属规则，仍保护 Torch、编译器和评测全局状态；KG 同样按已绑定 Catalog 的 evaluator 条件注入 Gems prompt。独立使用 Gems pytest 的复测工具没有自动接入该门禁。

Server 拒绝时 preflight 返回 FAILED、stage=candidate_admission、is_hack=true；evaluate/profile 直接执行各自的评测／分析职责，不对未经过 preflight 的直接请求提供准入拦截。复用已有 Protocol v6.2 字段，不新增 BLOCK 类别。通过结果仍可能含人工审核信号，不是任意 Python 无副作用的证明。

`/status.capabilities.candidate_admission` 声明 version=1、规则源码／黑名单 hash 和 stages=[preflight]。需要新准入能力的 KG 在提交前确认 capability，不根据 KGS release 推断能力。KG receipt 本地 schema 升至 3.0，绑定规则身份；PASSED+is_hack=true 不再签发 receipt，规则变更或旧 receipt 必须重新 preflight。本地 receipt schema 不是 Protocol version。KG 的 eval_round 保留 preflight receipt 和源码 hash 核对；直接调用 KGS 的用户自行保证已通过 preflight，Server 不增加新的签名凭证机制。

本版本仅检测已覆盖的显式源码模式，不声称阻止动态别名、任意文件写入或 ctypes 等所有间接绕过。运行中才能暴露的副作用、reference 错误与计时异常仍须相应执行证据；preflight smoke 不替代完整 correctness 和计时验收。

历史回放中的 `lift_fresh` 辅助 launch 与 `unsqueeze_` 额外 `squeeze_()` 不增加算子专用拒绝规则，避免将 preflight 扩展成逐算子语义检测器。既有最小 metadata 无 JIT 白名单保持不变；回放结果和未检出案例见 [历史候选源码门禁回放](candidate_admission_history_replay.md)。

## 验证

2026-09-07 验证实现为 KGS v6.3.1 开发分支 `feat/metadata-admission-policy@dc020d4`（基于已发布 v6.3.1，未发布新 tag），Protocol v6.2。统一准入与 FlagGems adapter 相关 host 测试 93 项通过，wheel 中已确认包含策略源码与 YAML 数据。目标通过 Gitee 临时验证分支取得同一 commit。

设备使用容器 `kernelgen-nvidia-cu128` 的物理 0 卡 NVIDIA A100-SXM4-40GB；容器实际镜像为 `kernelgen-nvidia-cu132-nsight2026.1:base`，Python 3.12.3、Torch 2.11.0+cu130、Triton 3.6.0、CUDA runtime 13.0、driver 580.126.20，未安装或更换依赖。loopback 临时 Server 配置一个设备 slot、两个请求线程、triton timing。

目标 Debug Job 中的 engine/native/admission/metadata 测试 50 项通过；两个白名单 recipe 的对象 identity、storage 与 shape 在 CUDA Tensor 上验证通过。`tests/live_server_validation.py` 对 `kernelgenbench_square` 完成 reference-as-solution、Triton preflight/evaluate、并发与普通 RUNTIME_ERROR 验证，36/36 workload 通过；故意退出 worker 的恢复测试产生一次 incident，一次强探针恢复，最后 slot idle 且 broken=0。

额外 HTTP 验证确认 Torch 全局修改与仅零 grid 候选在 preflight 和直接 evaluate 中拒绝，profile 也拒绝保护状态修改；候选顶层写入的导入标记不存在。该验证没有使用修改 Gems/Torch 或计时链的方式获得通过。

原始证据保存在验证机器的 `/data/akg_kernel_bench_lite/kernelgen/runs/candidate-admission-validation/`，包括 `debug-download/validation.json`、`live-server.json`、`admission-live.json`、`run_admission_validation.py` 和 Server audit/log。该路径是本地实验归档，不随 clone 分发。KG 配套的 receipt／Codex E2E 结果记录于其 `docs/operations/candidate_admission_validation.md`。

后续范围调整验证：限定 Gems 专属规则仅对 Gems adapter 生效，52 项相关 host 测试通过，覆盖 native 下不启用 Gems 规则、通用保护在两种 evaluator 下均生效，以及 Gems preflight／evaluate／profile 的导入前拒绝。本次未重新启动目标 Server，上述真机结果仍对应 dc020d4。

准入职责收敛后，b173f32 的 51 项 host 测试通过；三个依赖 Torch 的模块在现有 kernelgen-nvidia-cu128 容器、同 commit 的 Gitee checkout 中补跑 CPU 测试，24 项通过。CUDA_VISIBLE_DEVICES 为空，未启动目标 Server 或 GPU 实验，未安装包。正反用例确认 preflight 仍在导入前拒绝，而 evaluate/profile 不重复调用准入检测器；先前直接 evaluate/profile 拦截的测试结论仅保留为历史。原始记录在 KG 本地 runs/preflight-only-validation-20260907/。

## 2026-09-08 环境副作用规则补齐

候选模块在 import 和 run 路径均不得调用进程级 `empty_cache`，通过 `_set_allocator_settings`、`change_current_allocator`、`set_per_process_memory_fraction` 修改 Torch 公共显存分配策略，修改进程环境变量，直接写运行时文件或启动外部进程，也不得安装改变评测行为的注册 context。文件/进程规则使用通用 API 模式，避免按 shim 文件名逐个追加特例；Coder 编辑候选与独立诊断脚本、评测器自己的环境配置/缓存清理、已安装编译器正常写编译缓存不属于候选源码违规。

普通张量分配和仅通过普通分配提供必要 kernel workspace 的 `triton.set_allocator` 回调保持允许；它不替换 Torch allocator，不能仅凭 allocator 名称拒绝。回调内的缓存清理、Torch 策略修改和其他评测环境副作用仍按通用规则拒绝。此许可不证明注册作用域及设备、对齐、stream 处理正确，不新增 Server 自动 allocator 配置；合法工作区功能仍需在目标运行时验证。

规则覆盖明确的 Torch/厂商缓存 API、环境赋值/删除/映射变更、写模式 open、已识别 pathlib.Path 写操作及常见文件/外部命令 API。常见 import alias 和单次赋值 alias 能被识别；不确定的普通对象方法不以同名一概拒绝，动态构造和任意间接调用不具备完整保证。环境只读、环境副本更新、文件只读、元数据与正常张量分配保留允许。

Gems 的 override_registered_op/override_gems_op 入口仅在实际 FlagGems adapter 下拒绝，包括持久进入 context；不在 native 下启用 Gems 专属规则。源码检查仍仅由 preflight 在候选 import 前运行，evaluate/profile 不重复扫描。规则 hash 自动改变，既有 receipt 检查继续负责源码与服务策略身份。

本轮对应 KGS host 测试 101 项通过，包含四类新增违规在 adapter import/pytest 前拒绝、两种 evaluator 的通用规则和合法只读对照。历史 348 份源码回放从拒绝 20 份增至 30 份，21 条缺源码，SHA 全部吻合。旧“正常”标签 115 份中 5 份含新增禁止的真实缓存/allocator 副作用而被拒绝，其余 110 份通过；不重写旧标签，也不报告 115 份零误杀。新增拒绝不等于原表降速根因已闭环。

本次没有新增 GPU 验证或修改运行时，未部署新规则。证据位于 KG 本地 runs/preflight-environment-policy-20260908，配套说明位于 KG docs/operations/candidate_environment_admission_validation.md；这些本地实验输入须从授权归档恢复。
