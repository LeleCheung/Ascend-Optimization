# 候选准入：preflight 与 hack 检测统一

KG 通过 KGS preflight 完成候选源码准入、ABI 和目标编译 smoke。Server 的 native 与 FlagGems adapter 共用 `candidate_admission.py`，在 import 前拒绝明确违规；evaluate/profile 不重复执行准入或 hack 检测。KG 不维护第二套硬拒绝黑名单。

## 2026-09-08 确认的职责收敛与环境禁令

本节设计已落实到本分支的共享 Coder prompt，并在 KGS `feat/metadata-admission-policy` 补齐显式环境副作用规则。两个仓库的改动尚未合入主线或部署，不能把本地开发完成当作现有 Server 已启用新规则。

### Preflight 只保留两项职责

| 步骤 | 检查内容 | 手段 |
|---|---|---|
| 候选源码准入与通用 hack 检测 | 在 import 前拦截明确的计算 fallback、受保护状态修改和不满足实现契约的候选；遵守已有 metadata 无 JIT 白名单 | 静态 AST 与规则匹配，无法确定的接收者或间接行为只给审核信号 |
| 候选加载与最小执行 | 检查本次候选的加载、入口和参数契约，沿用已有 timing workload，各执行候选一次并同步，暴露编译、资源与运行错误 | 在目标环境实际导入、编译和调用，不执行完整精度比较或性能验收 |

Pytest review 在生成前运行一次原生 `--level core` reference-as-solution，重点检查 API、注入、编译与已确认的 reference 语义等 P0；workload/边界/性能路径等 P1 建议不阻塞。Preflight 不重复承担这套测试准备审查，但仍验证每一版新候选的加载与执行。数值正确性、性能是否达到阈值、reference 计时漂移与最终 best 独立确认，属于 Evaluate 和最终验收。

### Coder 与候选不得管理评测环境

Coder 负责候选算法、数值、访存、合法编译配置和性能优化。候选可正常管理自身临时张量；编译器、reference、注入和评测环境的问题由对应维护方修复。不得为了让候选通过而临时修改这些组件，即使事后恢复也不允许。

| 禁止行为 | 范围与理由 | 实现与边界 |
|---|---|---|
| 清理进程级显存缓存 | 候选 import 或调用期间不得调用 `empty_cache()` 等进程级缓存清理接口；这会改变后续 reference/其他 workload 的分配条件 | 已补齐显式调用静态拦截及 Coder prompt；评测器自己的统一缓存清理不受候选规则限制 |
| 修改 Torch 公共显存分配策略 | 禁止 `_set_allocator_settings`、`change_current_allocator`、`set_per_process_memory_fraction`；策略或限制变化可能影响 baseline 和后续 workload | 已有显式调用规则覆盖；普通张量分配和仅提供必要 kernel workspace 的 `triton.set_allocator` 回调不属于这类禁止行为 |
| 修改进程环境变量 | 禁止写入、删除或更新环境变量，包括改变编译器路径、开关与缓存配置；禁止修改后再恢复来绕过约束 | 已补齐明确的环境赋值、删除和变更调用；读取环境与修改环境分开处理 |
| 写入或替换编译器 shim、框架及测试代码 | 不得通过文件写入、替换、权限修改或外部命令安装 shim，改写编译器/JIT、Torch、reference、pytest 或 benchmark；Gems 专属保护按下述 evaluator 范围启用 | 通过候选内显式文件写入和外部进程调用规则拦截，不依赖 shim 文件名；任意间接写入仍不具备完整静态保证 |
| 改写注册或保留改变评测行为的 context | 禁止修改 Torch/评测注册与全局状态；Gems adapter 下禁止候选自行注册、替换测试接口或持久进入注册 context，例如 `override_registered_op(...).__enter__()` | 既有保护保留，已补齐 override_registered_op/override_gems_op 注册入口；不以任意对象的 `__enter__` 方法名一概拒绝 |

Gems 专属限制继续仅对实际 Gems adapter 启用，不能按算子名称触发，也不因这次补充扩展到所有 native 候选。Torch、编译器、环境变量和评测全局状态保护在 native 与 Gems adapter 下均适用。

环境读取、设备能力查询、tensor 的 shape/stride/dtype 查询、合法张量分配与释放、已有允许的元数据操作，以及 kernel 的合法 `num_warps`/`num_stages` 等配置保持允许。Coder 在授权 workspace 中编辑候选和编写独立复现脚本也保持允许；复现脚本用于诊断，不得用来修改受保护环境。提交的候选模块不直接写运行时文件或启动外部进程；已安装编译器自身正常生成编译缓存不属于候选源码写文件。Coder 正常使用编译器功能与安装/改写编译器 shim 是不同操作。

禁止 `empty_cache()` 是候选职责边界，不等于确认该调用是作弊或某次性能下降的原因。若测试需要缓存清理或存在碎片/OOM 问题，由评测维护方统一处理，保留原因与复现；不能让候选顺带改变下一次 baseline 的运行条件。

`triton.set_allocator` 注册的是 Triton kernel 临时工作区的分配回调，不是替换 Torch allocator，也不因 API 名字含 allocator 而拒绝。允许回调使用普通分配提供必要 workspace；回调内部仍受禁止缓存清理、Torch 策略修改和评测状态修改的通用规则约束。注册会保留运行时状态，允许这项用途不等于证明其作用域、设备、对齐或 stream 处理一定正确；目标运行时的 workspace 支持与隔离需由执行环境验证，本次不新增自动 allocator 配置机制。

### 失败处理与后续验证

明确的源码策略违规由 preflight 在导入前拒绝，沿用既有 `FAILED / candidate_admission / is_hack=true` 响应；Coder 删除违规行为后重新 preflight，不把策略违规上报为环境 BLOCK。普通加载、编译或执行失败保留阶段、源码和完整异常：候选语法、参数或 tile 资源问题交 Coder；疑似编译器问题按执行证据归因，不因“编译失败”自动判为编译器 bug。

后续只在 preflight 补齐通用规则，不在 evaluate/profile 增加重复扫描，不新增 `lift_fresh`、`unsqueeze_` 等逐算子语义黑名单。静态通过不证明所有动态副作用不存在，最小执行通过不证明精度正确或没有隐蔽越界。

共享 Coder implementation profile 已同步；验证要求：禁止行为及常见导入别名能够在候选 import 前拒绝；环境只读、元数据和正常分配不误杀；Gems 专属规则仍按 evaluator 生效；历史问题候选与正常对照分别回放。新增覆盖数量只报告实际回放结果，不能把规则新增等同于原表性能失败已经修复。

## 判定与职责

明确的框架计算回退、缺少要求的 JIT/launch、仅零 grid 的已识别 Triton launch、Torch/Gems/compiler/pytest 保护状态修改及已覆盖的注册／全局设置调用，在 Server 返回 candidate_admission 失败。普通同名对象方法等接收者类型不明的静态命中以 `ADMISSION_REVIEW` 保留，不自动定性为 hack。审核信号可随通过结果返回，仍需按审核流程核实，不表示数学或实现合规性已被完整证明。

Gems 专属 prompt 和注册／源码保护规则仅用于 evaluator=flaggems 的 Gems adapter。KG 按 Catalog manifest 的 evaluator 注入，已有 native snapshot 不重新解析；KGS 按实际创建的 adapter 类型启用，不接受候选自行选择，不按算子名或 framework 名推断。native 不注入 Gems 专属规则，仍受 Torch、编译器及评测全局状态保护。独立原生 pytest runner 不会因为 KGS 实现了门禁就自动获得保护，必须显式接入。pytest review 负责测试准备，候选准入负责候选边界，运行中 BLOCK 和计时验收保持各自职责。

维护者白名单允许 `lift_fresh` identity 与 `unsqueeze_` inplace view 的最小实现不使用 JIT；不豁免 ABI、alias、effects 和 correctness。不允许候选自行声明白名单，`resize_output_` 和 `unbind_copy` 当前不在其中。详细实现见 KGS `docs/metadata_admission_policy.md`。

Coder 的共同 implementation profile 明确禁止修改框架／reference／编译器／评测状态，以及用假 kernel 获得准入。准入失败后仅修复候选，再次 preflight；失败不占 measured round，不进入 BLOCK。现有硬门禁覆盖显式源码模式，不是任意 Python 的安全沙箱，不能证明动态间接写入或运行中的所有副作用均不存在。

## 能力与 receipt

`/status.capabilities.candidate_admission` 必须声明 version=1、64 位十六进制 policy_sha256 且 stages 包含 preflight。KG 在提交前检查能力，旧服务缺少能力时返回配置错误，不能静默继续。此版本是功能能力契约，不是 KG/KGS release 或 Protocol version；wire 继续使用既有 Protocol v6.2 字段。

KG 本地 receipt schema 改为 3.0，将 capability／规则 hash 纳入已有 service_signature。只有 status=PASSED 且未被标为 hack 才签发 receipt；PASSED+is_hack=true 明确失败。新 preflight 开始即撤销之前的 receipt，编译、通信或配置失败均不能留下旧通过记录。旧 schema、源代码／workload／服务／规则变化以及被拒绝的 target 记录都需要重新 preflight。未通过准入不得调用 eval_round，不向 ledger 伪造一次测量。KG 在 eval_round 核对 receipt 与当前源码 hash；直接调用 KGS evaluate/profile 的用户自行保证已完成 preflight，Server 不额外重复轻量检查或引入签名凭证。

本次不移动发布 tag，不改 KG/KGS release 编号；正式发布时按各自实际变更管理版本，选择支持上述 capability 的组合。旧 campaign 不自动升级 KGS 或 framework；新门禁使用独立验证 workspace。

实际结果见 [A100 候选准入验证](../validation/candidate_admission_validation.md)。

本轮 host 验证与历史回放见[环境副作用准入验证](../validation/candidate_environment_admission_validation.md)。
