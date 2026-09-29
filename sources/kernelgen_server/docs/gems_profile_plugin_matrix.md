# 显式 pytest 插件：fresh-master 真机回归

## 配套与边界

本轮使用 KGS v6.3.4 开发快照 `a0b6bd2cb4e085209d821dc4ed9ff4d7144673d4` / Gems `codex/pytest-profile-plugin@24823b6695c5a2651e2dc0745e2f265d0c40310e` / Protocol v6.2。Gems 从 master `488a91030a9465ada6dcc6104526da00d8c7057d` 创建新分支，只迁移显式 pytest 插件修改；已合并的 #6621 不包含这项修改。KGS 的 a0b6 相对 a8a7 只有文档与 Gems 配套清单变化，Profile 运行时代码相同。

这轮按用户要求只测试真实 pytest 和 KGS API，不重跑模型、Coder、KernelGen Workflow 或 CLI 后台 E2E。使用原始 `negative` pytest：15 个 Preflight case、18 个 correctness case、15 个 timing case。候选是有界 grid-stride 的纯 Triton negative，不修改 reference、测例或覆盖门禁。

每个平台使用一个临时设备 slot、4 个请求线程；远端 KGS 仅监听 loopback，本地经 SSH stdio HTTP proxy 访问。测试保留 Preflight 正例、运行期抛错负例、完整 Eval、所有声明支持的 Profile level、capture 中抛错负例及失败后的再次采集。正例要求 normalized metrics 中存在候选 kernel，并有非空的非日志产物；负例不能伪造 completion marker；最后检查 slot 空闲健康。

## 测试记录

本地证据位于 KG 工作区 `runs/gems-profile-plugin-followup-20260923/`。`validate_matrix.py` 保留原始 API 响应，`collect_artifacts.py` 下载产物并记录 SHA256，`summarize.py` 从响应生成 `summary.json`，`cleanup.py` 校验 PID 启动身份、确认 idle 后关闭本次临时实例。旧实验及失败尝试没有被覆盖。

八个平台均完成本轮检查。下表来自 `summary.json`；所有平台的 Preflight 运行期异常均返回 `RUNTIME_ERROR`，capture 内异常均返回 `failed`，随后再次采集均 `completed`，最终 scheduler 的 active/waiting/checking/broken/incidents 均为 0，available=healthy=device_slots=1。昆仑芯按用户此前确认暂缓，不计入通过数。

| 平台 | 物理卡 | Preflight | 完整 Eval（正确性 + timing） | metrics / instruction | 失败后再次采集 |
| --- | --- | --- | --- | --- | --- |
| NVIDIA A100 | 0 | 15/15 | 33/33 | completed / completed | completed |
| 海光 | 6 | 15/15 | 33/33 | completed / completed | completed |
| 摩尔线程 | 7 | 15/15 | 33/33 | completed / 未声明支持 | completed |
| 沐曦（显式 vendor） | 7 | 15/15 | 33/33 | completed / completed | completed |
| 昇腾 10.0.0.9 | 7 | 15/15 | 33/33 | completed / completed | completed |
| 平头哥（NAS） | 15 | 15/15 | 33/33 | completed / completed | completed |
| 燧原 | 7 | 15/15 | 33/33 | completed / completed | completed |
| 天数 | 14 | 15/15 | 33/33 | completed / completed | completed |

共 23 次成功 Profile 和 8 次预期失败 Profile。成功报告已逐项下载并校验字节长度、记录 SHA256；不是只检查 HTTP 200 或顶层 completed。临时实例在保存报告及 Server 日志后关闭，不留作长期部署。

## 沐曦显式 vendor 配置

Gems master 在后端自动探测中可能导入 `_kunlunxin`，该模块的 `_install_register_config_patch()` 在 import 时修改共享 `GeneralOpRegistrar.__init__`。沐曦自动探测路径触发该副作用，随后 `use_gems()` 会进入昆仑芯 `_extend_config()`，报 `ModuleNotFoundError: No module named 'triton.language.extra.xpu'`。18 个正确性 case 在调用候选前失败，因此 KGS 的候选覆盖检查拒绝 Eval；不是 Profile 插件丢失候选。

本轮使用 Gems 已有的 `GEMS_VENDOR=metax` 显式指定真实后端，避免枚举其他厂商模块。诊断性直接运行原始 pytest 后 18/18 通过，再使用相同配置重启本次临时 KGS 完整复验。不删除 pytest、不跳过候选覆盖检查、不升级 Torch/Triton、不在目标机补丁修改 Gems 源码。该配置是沐曦本轮验收条件，不能将结果描述成 master 默认自动探测已修复。原失败保留在 `metax-autodetect-validation.json`、`metax-autodetection-diagnosis.json` 和 `metax-autodetect-server.log`。

## 环境及验收限制

平头哥的源码、临时目录、缓存和产物均位于 NAS `/mnt/workspace`，不使用故障 EXT4 分区；通过不表示根分区已修复。天数使用已验证可用的物理卡 14；昇腾仅使用 inventory 中的 10.0.0.9。昇腾首次 clone 已得到正确且干净的两个 checkout，重复 fetch 的 TLS 错误不影响已验证的 exact commit；部署证据保留这一失败，不改用其他代码。

Gems 新 worktree 的三组回归为 73 passed；KGS adapter/runner/Hygon Profile 回归为 101 passed；KG Server 管理及 Gems 安装入口回归为 102 passed。执行前均核对实际导入路径。宿主 Python 缺 Torch，KGS 的 HTTP ALL_SKIP 单测首次因此失败；转到已有 A100 Python 环境后同组 101 项全部通过，没有安装依赖。没有变更 Torch、Triton 或厂商运行时，也没有新建模型实验。

这些测试覆盖插件接线、原始 pytest 回归及成功/失败资源释放，不是全 Gems 算子或所有 pytest 插件组合的穷尽验证，不把功能通过当作性能提升结论。

## 后续：由 KGS backend 派生 vendor

KGS `376ae33600bc42b5a5f831272b688e438c280123` 将 vendor 选择放回 Gems adapter：inspect、Preflight、Eval 和 Profile 共用 `_base_env()`，根据 Server 已选 backend 设置子进程 `GEMS_VENDOR`。名称转换复用 Profile runner 的映射，不维护两份表；不再次探测硬件，不修改 Server 全局环境，也不增加 CLI 参数。Gems 源码未变，仍为 `24823b66`。

相关回归 123 passed，覆盖全部十一种 backend 的名称映射、未设置/错误继承的环境变量、全局环境不变以及 Profile 命令传递。沐曦真机复验单独保存在 KG `runs/gems-vendor-from-backend-20260923/`，清除启动环境中的四个 vendor/backend 选项，并核对 `/proc/<KGS pid>/environ` 中它们不存在；不将上一轮手动配置的结果视为本修复通过的证据。

沐曦复验通过：inspect 成功、Preflight 15/15、完整 Eval 33/33、metrics/instruction 均 completed；运行期失败的 Preflight 返回 RUNTIME_ERROR、capture 内异常返回 failed，随后再次 metrics 采集 completed。最终设备 slot 空闲健康，incidents=0。KGS 集成不再需要用户手动设置 `GEMS_VENDOR`；Gems 独立运行时的自动探测副作用仍是上游问题。本修复没有重复八平台模型 E2E，其他 backend 的映射由上述回归测试覆盖。
