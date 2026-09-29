# Gems pytest 多 Coder KernelGen 验证

本报告记录 `codex/gems-pytest-integration` 的开发验收，不是软件发布。最终确认：除用户明确暂缓的昆仑芯外，八个平台的真实 CLI 后台任务均为 SUCCEEDED，完成两个 epoch、四个 measured round，各轮均为 PASSED 33/33；每个平台均有真实 completed 的 profiling 返回和有效报告，满足用户最终确认的链路验收口径。该口径不要求每个 Coder/每一轮都完成 Profile，也不把 CLI 成功解释为全部子任务复验成功或性能全部达标。历史取消、NEEDS_RETEST 和 pending 记录保留，未改写。

## 改动边界

移除 Catalog 输入和 KernelGen preparation 中两处 Gems 仅限 SimpleOpt 的限制。Gems 继续使用远端导出的不可变 evaluation snapshot、pytest case ID、benchmark fingerprint 和原始 correctness/benchmark；不复制 Native workload，也不另写优化循环。后续 epoch 复用同一输入契约，变更 fingerprint 时仍拒绝续跑。

KG 主工作区先更新到 `dev@d67bea6f`，本地原有修改保留。功能验证使用 `KG v6.5.1` 开发分支 `cff4e027` / `KGS v6.3.4` 开发分支 `5c3bb659` / `Protocol v6.2`；这些 commit 不是同名 release tag 的原内容。Gems 使用 `feat/kernelgen-profile-hook@c77cf9c7`，基于当时 master `54b28861`。KGS 后续 `d5dae6b8` 补齐共享 Profile 编译器准备，在燧原验证；其他正在运行的 campaign 不自动升级。Gems `96f6cf77` 只追加格式和许可证头整理，单独重跑了 69 项回归。

## 配置与证据

算子为安装式 `flaggems-adapter-definitions/negative`，原始 pytest 共 18 个 correctness case 和 15 个 timing case。每个平台一个临时设备 slot，2 Coders × 2 epochs，每个 Coder 每 epoch 1 round，开启 Profile，Runtime timeout 1800 秒。模型为 `deepseek-v4-flash[1m]`。A100 从零生成；远端以可移植的纯 Triton negative 实现作为待验证 seed，每个目标重新执行原 pytest，绝不复用 A100 的计时作为目标事实。这是功能 smoke，不是充分优化或全算子覆盖结论。

证据目录为 KG 主工作区 `runs/gems-kernelgen-e2e-20260923/`，包含提交 workspace、ledger、epoch completion、最终 JSON、CLI status/history、服务器 audit、Profile 报告和部署记录。`summarize.py` 从 ledger 与最终 JSON 汇总；`observe_cli.py` 保存 CLI 只读观察。远端参数从当时的 `tests/multi_device_batch_hosts.conf`（现已并入 `tests/hosts.md`）读取，KGS 只监听 loopback，本地经 SSH stdio proxy 访问。未更改 Torch、Triton 或厂商运行时。

示例提交（需使用本 feature 的 KG，endpoint 替换为已经 ready 的临时 KGS）：

```bash
kg run --mode kernelgen --catalog-name flaggems-adapter-definitions \
  --definition negative --eval-server http://127.0.0.1:25202 \
  --workspace /path/to/new-workspace --n-parallel 2 --n-epoch 2 \
  --min-rounds 1 --max-round 1 --profile --timeout 1800
kg status /path/to/new-workspace --detail
kg history /path/to/new-workspace --json
```

## 已完成的验收

- A100 / `kernelgen-nvidia-cu128` / 物理卡 7：`a100-negative-02` 为 `SUCCEEDED`，4/4 阶段完成；四个 measured round 均 `PASSED 33/33`，四次 Profile analysis 为 `completed`，两个 epoch completion 均落盘。最终 KernelGen 输出为 `PASSED`；精确加速比由 `summary-observed.json` 汇总。history 记录候选最初 measured round 的结果，最终输出记录其复验结果，二者是不同执行，不能互换。临时 KGS 8021 已在 idle 后关闭。
- CLI `status --detail` 返回 schema 2.0 进度；普通阶段为 `progress_kind=basic, progress={}`，优化阶段为 epochs 进度；history 返回四条 ledger series，没有伪造 Native 测例。
- 八个平台的最终 CLI 任务均 SUCCEEDED，两个 epoch completion 和最终输出均落盘；每个平台均有成功 Profile 返回及非空报告。天数使用物理卡 14，平头哥使用物理卡 15 与 NAS。最终性能及 Profile 子项状态见下表，不以顶层成功覆盖历史异常。
- 燧原 `d5dae6b8`：Gems metrics 和 instruction `/profile` 均 `completed`，warmup 1 / iterations 2 下报告恰好两次候选调用；instruction 生成设备二进制、IR、反汇编、源码映射与报告。Native 和 Gems 在编译前共用 line-info 准备逻辑；不是通过取消 marker 或范围校验获得通过。
- KG：宿主机 106 passed / 1 skipped（缺 Torch），在已有 A100 环境重跑同一套为 107 passed。Gems：69 passed，black/isort/flake8 通过。KGS：Gems adapter/runner/Hygon Profile 相关 95 passed。测试前均核对实际导入路径来自配套 feature worktree。

## 最终汇总（2026-09-23）

此表由各 workspace 的最终输出、ledger 和 Profile manifest 生成；加速比取最终复验，不取搜索阶段峰值。所有行均完成 4 个 PASSED 33/33 的 measured round，CLI 为 SUCCEEDED。

| 平台 | 最终复验加速比 | completed 且有有效产物的报告数 | Ledger Profile 状态 |
| --- | ---: | ---: | --- |
| NVIDIA A100 | 1.010× | 25 | completed=4 |
| 海光 | 1.053× | 30 | completed=4 |
| 摩尔 | 0.996× | 10 | pending=2；completed=2 |
| 沐曦 | 1.003× | 14 | failed=2；completed=2 |
| 燧原 | 0.094× | 18 | completed=3；pending=1 |
| 昇腾 | 0.541× | 20 | completed=4 |
| 平头哥（NAS） | 1.086× | 32 | completed=4 |
| 天数（卡 14） | 1.047× | 34 | completed=4 |

昇腾和燧原低于实验性能合格线 0.8×；本次只验收功能链路，不将它们称为性能达标。昆仑芯未纳入通过数，按用户确认暂缓。

## 失败证据与尚未完成

1. 首次 A100 提交发生在 KGS readiness 之前，留下 `a100-negative` 的 0 round 失败；没有 lifecycle 的任务不能直接 resume。保留现场后在 Server ready 时另建 `a100-negative-02`，未覆盖失败目录。
2. 燧原直接复用 A100 候选时，最大测试 shape 的 grid.x=524288 超过平台上限 65535。换用有界 grid-stride 的待测代码后通过；原始 pytest/workload/reference 未变。
3. 昇腾第一次完整 Eval 的 240 秒上限不足，返回 `TIMEOUT`；超时强探针通过、slot 恢复。在保留原证据后使用更长上限复测通过 33/33，没有把 timeout 当作 correctness 通过。
4. 天数最初只筛选无常驻进程的卡，0、8、10、11 均未通过最小 context/计算验证，卡 11 的 KGS 启动探针超时。该范围不能代表整机都不可用。用户授权检查其余卡后，1～7、12～15 的最小 PyTorch 计算全部通过，9 超时。后续选择物理卡 14，保留原常驻进程，使用 KGS `d5dae6b8` 启动独立 loopback 24201（SSH stdio proxy 25201）、4 个请求线程和 1 个设备 slot；强探针、Gems Preflight 15/15、完整 Eval 33/33 均通过，继续真实 CLI KernelGen 验证。没有绕过探针或重置设备。
5. 昆仑芯 KGS 启动探针通过，但 Gems backend 在 import 时调用旧 Torch 2.5 不支持的 `Library.impl(..., allow_override=True)`，无法生成 case list。对比 Gems `9623276` 和 `c77cf9c7`，该调用没有变化，并非此次同步新引入的差异。诊断性适配参数后，还出现 `_kunlunxin.ops.range` 与 `flag_gems.runtime.backend._kunlunxin.ops.range` 重复加载导致重复注册；仅删除参数不足以修复。没有升级受保护 Torch 或跳过 vendor 注册用于验收。完整 Workflow 阻塞，旧 xprofiler 问题也未解除。
6. A100 epoch synthesis 曾调用未初始化的 Knowledge 查询上下文并得到结构化错误，随后仍完成 synthesis。这是额外观察，不把本次 smoke 扩大解释为全部 Knowledge 查询路径已验证。
7. 平头哥初始 Preflight/Eval 通过后，模型候选编译反复遇到 Triton 缓存 JSON 解析错误。共享缓存有 12 个内容全为 NUL 的 JSON、磁盘及 inode 空间足够；创建独立状态目录返回 `OSError: [Errno 117] Structure needs cleaning`。内核进一步确认根分区 `/dev/vdr` 的 EXT4 分配块与文件系统元数据重叠、延迟分配失败，并提示 `Data will be lost`。实验目录和 `/tmp` 在同一文件系统，不能把换到 `/tmp` 视为换盘修复。已协作式取消 campaign、关闭本次临时 KGS 和 proxy、回收证据，没有删除共享缓存或自动修复文件系统。不能把此前 API 通过扩大为完整 Workflow 通过。
8. 初始实例只有一个请求线程，长 Eval/Profile 与元数据查询共用 executor。核对客户端调用后，60 秒超时对应 `/operator-contract`（不是 `/inspect` 的默认超时）。沐曦协作式取消释放 Profile、强探针恢复后，保留代码/pytest/workspace，重启本次实例为 4 个请求线程、1 个设备 slot，再续跑成功；第一 epoch 两条 Profile failed 保留取消原因。Coder 并发仍为 2，没有增加同卡 Eval 并发，该配置限制不同于 SSH stream 或设备故障。
9. 摩尔 `1R/agent1`、`2R/agent0`，燧原 `1R/agent1` 的最终复验或契约复查遇到上述 60 秒超时，子任务为 NEEDS_RETEST。SingleCoderOptimizationWorkflow 在复验未通过时提前返回，没有进入最终 best 的兜底 Profile，ledger 因而保留 pending。它们不是 profiler 已执行失败，也不是未成为全局 best 而按策略跳过。KernelGen 的 best 选择排除未复验确认的结果，因此其他已确认候选仍可让顶层 SUCCEEDED。昇腾也保留一条契约复查超时的子任务记录。按用户最终确认的验收口径，不为凑齐 Profile 次数补跑或伪造 completed；状态收尾与控制请求线程隔离仍是后续改进项。

### 平头哥 NAS 复测

用户授权后在 `/mnt/workspace` 的 `fuse.aliyun-alinas-efc` NAS 建立独立测试目录 `gems-kernelgen-1rd265lw`，该挂载与故障根分区不同。为隔离存储变量，保持原 PPU campaign 的 KGS `5c3bb659` / Gems `c77cf9c7` 和原解释器，不夹带升级；KGS 通过 Gitee 重新 checkout。源码、TMPDIR、Triton/Torch 编译缓存、XDG cache 和 state/report/audit 都放在 NAS，禁用 Python 字节码写入。原 EXT4 checkout、缓存和失败现场未修改。

原 agent0、agent1 两份候选的源码 SHA256 和 benchmark fingerprint 与旧失败记录一致；二者均通过 Preflight 15/15、Eval 33/33。NAS 主缓存的 30 个 JSON 全部可解析。原 agent0 候选的 ACU metrics 和 instruction `/profile` 均 completed，instruction 具有二进制、反汇编、instruction listing 和源码映射产物。此后 `kg resume` 复用原 `ppu-negative` workspace，继续 2 Coders × 2 epochs 的真实 CLI 流程，未把外部 API smoke 结果伪造为 Coder ledger。

这说明当前测试可在 NAS 继续，不代表 EXT4 元数据错误已修复。证据为实验目录下 `ppu-nas-deployment.json`、`ppu-nas-retest.json`、`ppu-nas-cache-check-job.json`、`ppu-profile-metrics.json`、`ppu-profile-instruction.json`，完整诊断过程保存在同目录 `BLOCKER_DIAGNOSIS.md`。

临时 lint venv 新增 black 26.5.1、isort 5.12.0、flake8 7.4.0 及其依赖，仅用于本地主机代码检查，没有安装到芯片运行环境。其余远端临时 Server 在 campaign 结束、slot idle 并回收证据后关闭；不要将正在运行的临时实例当作长期部署。

## 后续：显式 pytest 插件接线

原 PR 内继续将 profiling hook 改为 `pytest.main(..., plugins=[plugin])`，删除环境变量模块发现、动态 import、`__main__` 约定及 KGS 全局 hook 转发。Gems 声明单一 `pytest_flaggems_profile_scope(backend, case_id)`，KGS 提供返回 context manager 的插件；输入、预热和迭代仍由 Gems 负责。backend/case 校验、异常时清理和恰好一次成功 capture 的 completion marker 规则不变。

初次插件验证时 KG 锁定 KGS `a8a7f4c9f3d346617d60b7d57f658236ee81375d`，其 compatibility.yaml 固定 Gems `79ab71d17b7f5437bdd64185cf570c12c36bf630`，Protocol 保持 v6.2。新接线回归：KG 107 passed、KGS 101 passed、Gems 73 passed。A100 与昇腾 10.0.0.9 真实 `/profile` 正向均 completed 并有候选 kernel 的报告，capture 内故意失败时均 failed、未伪造成功 marker，结束后 slot 空闲健康；A100 另有 Preflight 15/15、Eval 33/33 和独立无插件重放验证。

这不是八平台 E2E 的重跑：上面的完整多 epoch 表格仍属于先前记录的旧桥接版本。新接口实验单独保存在 `runs/gems-profile-plugin-20260923/`；对应设计与验证见 KGS 的 `docs/gems_profile_plugin.md`。本轮不改模型提示词、优化算法、HTTP Schema 或软件 release/tag。

## Fresh-master 插件复验与最终配套

Gems #6621 合并时只包含旧环境变量桥接。遵照分支关闭规则，插件修改迁移到基于 master `488a9103` 的新分支 `codex/pytest-profile-plugin@24823b66`，并在八平台测试完成后提交 [Gems #6628](https://github.com/flagos-ai/FlagGems/pull/6628)。KG 锁定更新为 KGS `a0b6bd2cb4e085209d821dc4ed9ff4d7144673d4`，由该 commit 的 compatibility.yaml 固定新的 Gems exact commit；KG v6.5.1、KGS v6.3.4、Protocol v6.2 的版本标签均未移动或重新发布。

按用户要求没有重跑模型/Workflow E2E，而是八个平台分别运行真实原始 pytest 和 KGS API：A100、海光、摩尔线程、沐曦、昇腾 10.0.0.9、平头哥、燧原、天数全部通过 Preflight 15/15、完整 Eval 33/33、声明支持的 Profile 级别、故意失败后的再次采集与设备释放检查。共 23 次成功采集、8 次预期失败，产物已下载并记录哈希，临时 KGS 和 proxy 全部关闭。Gems 73、KGS 101、KG Server 管理相关 102 项回归通过。

上述 a0b6 配套验证中，沐曦需显式设置 `GEMS_VENDOR=metax`：当前 Gems master 的自动探测导入昆仑芯模块会污染共享 registrar，导致正确性测试在候选调用前报缺失 XPU 模块。使用已有 vendor 配置后原始 pytest 和完整 API 矩阵通过，不代表该上游默认探测问题已修复。平头哥继续使用 NAS，原 EXT4 故障未修复。昆仑芯按用户确认暂缓，不计入通过数。没有改动 pytest 测例、候选覆盖门禁或受保护运行时。

新证据目录为 `runs/gems-profile-plugin-followup-20260923/`，原失败及旧实验记录保留。逐平台配置、完整结果和限制见 [KGS fresh-master 验证报告](https://gitee.com/BaaiAC/kernelgen_server/blob/676005fbb86d00c8e5cbf2f43e9b41651cd173ea/docs/gems_profile_plugin_matrix.md)。这些结果验证插件集成，不是所有 Gems 算子或性能优化的穷尽验收。

### 后续自动 vendor 配置

KG 当前锁定进一步更新为 KGS `376ae33600bc42b5a5f831272b688e438c280123`，Gems exact commit 不变。该版本由 Gems adapter 按 Server backend 自动设置子进程 `GEMS_VENDOR`，和 Profile runner 共用一份名称映射；用户不再需要上述沐曦手动配置，也不新增 CLI 参数。只修改子进程环境，不改变 Server 的全局环境或再次探测硬件。

清除全部 vendor/backend 启动环境变量后，沐曦真实 inspect、Preflight 15/15、Eval 33/33、metrics/instruction Profile、故意失败及失败后的再次采集全部通过，slot 空闲健康。证据单独保存在 `runs/gems-vendor-from-backend-20260923/`。KGS 相关回归 123 passed，KG Server 管理回归 102 passed；没有重跑模型/Workflow E2E，也没有将其他平台上一轮结果重新标注为本提交的真机结果。Gems 独立运行的默认探测副作用不在此处修复。
