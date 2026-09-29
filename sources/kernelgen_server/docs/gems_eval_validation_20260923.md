# Gems 新 override 接口的完整 Eval 接线验证

2026-09-23，KGS `feat/flaggems-profile-hook@5c3bb65988955b25025be3783f07a39d7b08640a` / FlagGems `feat/kernelgen-profile-hook@c77cf9c71285b8a0799bbac8693abb49cfc72fc6` 已在 A100 完成真实 `/evaluate` 正向与负向验证。Gems 分支已同步至当时 `origin/master@54b28861`。Protocol 仍为 v6.2；这不是发布新 tag，也不是全芯片 KernelGen Workflow 验收。

这是[Preflight 验证](gems_preflight_validation_20260923.md)之后的接线工作：Eval 不再使用 `--candidate-code-path` / `--candidate-report-path` 或独立 `candidate_coverage.json`，而是与 Preflight/Profile 共用 `--override` 参数构造。

## 事实与行为

- 正确性 pytest 在原结果 JSON 的每个 case 中写入 `candidate_calls`，记录测试调用阶段内真实触发的注入候选次数；KGS 要求所有非 skipped case 都有目标算子的正整数调用记录。
- benchmark 原生 baseline 仍单独计时，候选路径解析到 live override；只有该候选在计时阶段确实被调用后才写入 `candidate_source: override`。未调用候选不能只凭 pytest 通过就视为评测成功。
- 全部 correctness case skip 时保留 `ALL_SKIP` 并继续适用的 benchmark；空报告、未注入候选、缺失/重复 timing case 不能伪装通过。
- 候选在 benchmark 中报错可能导致 pytest 提前结束、没有完整 timing case 报告。这种非零退出返回结构化 `RUNTIME_ERROR`，不再因为后续覆盖检查而变成 HTTP 500；退出码为零却报告不完整仍按契约错误处理。

## 验证

在既有 `kernelgen-nvidia-cu128` 容器中使用空闲物理卡 7 和独立 loopback `8019` 临时 KGS，测试 `addmm_`。正向 candidate 调用 Gems Triton `addmm` 并更新输入，负向 candidate 显式抛出 `RuntimeError("intentional-eval-error")`。没有安装或升级 Torch、Triton 或厂商运行时。

| 验证 | 结果 |
| --- | --- |
| 原始 correctness pytest | 36 passed；每个 case 均记录实际候选调用 |
| 原始 benchmark pytest | 15 个 timing case，均为 override，baseline/candidate latency 有效 |
| KGS `/evaluate` 正向 | PASSED，51/51 workloads，具有有效几何平均加速比 |
| KGS `/evaluate` 负向 | HTTP 200 / RUNTIME_ERROR，无虚构的性能结果 |
| Gems 相关回归 | 69 passed；同步 master 后重新验证通过 |
| KGS 相关回归 | 100 passed |
| KG 更新依赖后的安装/Server/参数回归 | 105 passed；隔离导入最新 KG feature worktree 与本 KGS |

上述加速比只是 reference-as-candidate 的链路检查，不是模型优化结论。请求完成后 scheduler 为 `active=waiting=broken=0, available=1`，临时 KGS 已停止，旧服务未变。证据保存在 KG 工作区 `runs/gems-kernelgen-e2e-20260923/`，包括前后结果、原始 pytest JSON、HTTP 请求/响应、status 和 request audit；修复前的 HTTP 500 也单独保留。

## 配套 Workflow 验收

KG 主工作区已 fast-forward 到 `dev@d67bea6f`，原未提交内容和 index 均已恢复核对。KG 配套锁定更新在独立 `codex/gems-pytest-integration` 工作区准备，尚未发布。

用户随后确认开放 Gems 的多 Coder `--mode kernelgen`。KG `codex/gems-pytest-integration@cff4e027` 已移除两处 SimpleOpt-only 门禁，保留 snapshot/fingerprint 校验；A100 真实后台任务 `a100-negative-02` 已以 2 Coders × 2 epochs × 1 round、Profile 开启运行至 `SUCCEEDED`，四个 measured round 全为 `PASSED 33/33`、四次 Profile analysis 均完成，两个 epoch completion 和最终输出均落盘。原始算子为 Gems `negative`，没有抽取 Native Catalog。完整证据与仍在执行的平台记录由 KG `docs/validation/gems_kernelgen_20260923.md` 维护，不能用本次 A100 结果代替全芯片验收。

后续八个平台（A100、海光、摩尔、沐曦、昇腾、平头哥、燧原、天数）的真实 KernelGen CLI 均已 SUCCEEDED，各自完成 2 Coders × 2 epochs × 1 round，四个 measured round 均为 PASSED 33/33；每个平台均有 completed 的 profiling 返回和有效厂商报告。用户确认以“至少一次真实成功调用及有效报告”验收 profiling，不要求每个 Coder/每一轮都完成。历史取消、超时、NEEDS_RETEST/pending 保留，不改写为通过。KG 最终明细与性能结果由 `docs/validation/gems_kernelgen_20260923.md` 维护；功能链路验收不等于性能全部达标。

天数最初选中的卡启动探针失败，后在用户授权检查剩余卡后选物理卡 14 完成；平头哥原 EXT4 分区有元数据分配错误，改用独立 NAS 放源码、临时目录、缓存与报告后完成，未修复原分区或更改运行时包。昆仑芯仍在导入 Gems 时因旧 Torch 不支持 `allow_override` 和 vendor 模块重复加载阻塞，用户确认暂缓，不声称它已通过；该参数在此前 Gems `9623276` 中已存在，不是此次同步新增。旧 XProfiler 超时也仍未解除。

摩尔、燧原部分子任务的 pending 来自最终复验阶段 `/operator-contract` 60 秒超时后的提前返回。初始一个请求线程与长 Eval/Profile 共用 executor 会使控制查询排队；沐曦调整为 4 个请求线程、保留 1 个设备 slot 后续跑。未复验确认的结果不进入 KernelGen best 选择；此次不额外修改请求池或状态 Schema。

## Profile runner 配套复验

`d5dae6b8` 将 Native runner 已有的燧原 line-info 编译器准备逻辑移到 Native/Gems 共用入口，并在 pytest 导入 candidate 前执行，避免 Gems 的 instruction Profile 缺少前置准备。95 项 Gems runner/adapter/Hygon 相关回归在现有 A100 环境通过，实际 import 来自本 feature；宿主机无 Torch 的单项 API 测试另在容器补验，不因依赖缺失修改生产逻辑。

燧原临时实例通过 Gitee 更新到该 commit 后，以同一 Gems `negative` case、warmup 1 / iterations 2 分别请求 metrics 和 instruction。两者均 `completed`，报告恰好记录两次候选调用；instruction 产出设备二进制、编译 IR、反汇编、instruction listing 和 source mapping。这是对新公共准备逻辑的真机验证，未更改原始 pytest、参考实现或运行时依赖。
