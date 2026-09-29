# Gems candidate-only Preflight 验证（2026-09-23）

后续完整 Eval 的 override 接线与 A100 验证见[Eval 验证记录](gems_eval_validation_20260923.md)。下文保留本次 Preflight 阶段的原始版本与范围。

结论：Gems `feat/kernelgen-profile-hook@4175588eea31003288833e5cb9013f5545ee301c` 已实现 `--preflight-only`；配套 KGS `feat/flaggems-profile-hook@0bc1bf970848ea15548ae56fbe48dbb5af8b8e10` 的 `/preflight` 已改用 `--override` 和逐 case 报告。A100、海光的真实 HTTP Preflight、失败负对照和 Profile 回归均通过。本次没有迁移 `/evaluate` 的旧 candidate-code/report 接口，不能将这些结果表述为完整 Eval 或优化 Workflow E2E 通过。

## 语义与接线

Gems 沿用原 benchmark 的 case 枚举和输入构造，每个选中 case 调用一次候选并同步设备；不运行 correctness reference、不主动 warmup/benchmark、不计算 latency/speedup。候选内部的编译/autotune 仍可能执行多个设备 kernel，因此“一次”指 Python 算子入口调用次数。无 `--case-id` 时检查全部枚举 case；可重复传入该参数选择子集。未支持 case 枚举的 benchmark 明确失败，不回退完整 benchmark。

```bash
python -m pytest -q benchmark/test_addmm_.py --level core \
  --preflight-only --override addmm_:/path/candidate.py:run \
  --record json --output /tmp/preflight.json
```

以上命令从 Gems checkout 根目录执行。可执行性通过不等于数值正确或性能达标。`--preflight-only` 与 profile-only/list-cases/query/非零 parallel 互斥；候选异常、空 case 计划、未知/未执行的显式 case 和全部 skip 都不能算成功。

Gems 报告使用独立的 `flaggems.preflight/v1`，每次覆盖旧文件。KGS 同时检查退出码、报告 schema、operator、override 确实启用、各 case 为 passed、case ID 全覆盖且不重复、每个入口恰好调用一次。原 Profile 与 Preflight 共用 override 参数构造，Gems 两种模式共用候选选择与 dispatch；没有增加旧参数的兼容别名，也没有重新实现 case 生成器。Protocol 仍为 v6.2，软件 release 不因本次开发提交被移动。

## 验证结果

| 项目 | A100 | 海光 |
| --- | --- | --- |
| Gems `addmm_` 全部 core case | CLI 与 KGS 均通过 15 个 case | KGS 通过 15 个 case |
| 候选主动抛出 RuntimeError | CLI 退出 1；KGS 返回 RUNTIME_ERROR | KGS 返回 RUNTIME_ERROR |
| 既有 `/profile` 回归 | completed / NCU | completed / hipprof，PMC 记录正常 |
| 请求结束后 scheduler | active=waiting=broken=checking=0，available=1 | 同左 |

A100 还实测了 `--list-cases` 与 Preflight 15 个 case ID 完全一致、单 case 重放、未知 case 拒绝、互斥模式拒绝和 direct Gems profile runner completion marker。相关单元测试为 Gems **66 passed**、KGS **95 passed**，覆盖无参考/计时调用、逐 case 执行、构造/候选/同步异常、空计划、失败/skip/中断、旧 JSON 不混入、报告格式、未注入候选以及缺失/重复/非法调用次数。测试前核对导入路径为对应 feature worktree，没有使用主目录 editable install 代替被测实现。

实测环境：

- A100：本地 `kernelgen-nvidia-cu128` 容器，物理卡 7，容器既有 Python，临时 KGS loopback `8018`；请求与产物保存在 KG 工作区的 `runs/gems-preflight-20260923/a100/` 和同级 `a100-api-*.json`。
- 海光：inventory 的 `10.232.2.26` / `codex_fib_hygon_20260728`，物理卡 6，解释器 `/workspace/kernelgen_e2e_20260728/deployments/stage2-kg631-EdHzTP/venv/bin/python`；临时 KGS loopback `24123`，本地经项目 SSH stdio proxy 的 `25123` 访问。独立 checkout 与现场位于 `/workspace/kernelgen_e2e_20260728/gems-preflight-HGKPgi/`，未修改共享 Gems checkout。
- KGS 使用 v6.3.4 开发快照 / Protocol v6.2；两个 exact commit 如本文开头。没有安装、升级或替换 Torch、Triton 或任何厂商运行时。

Profile ID：A100 `fed73ce85fe6488eb58361519a88f807`，海光 `04eff632d5364632a6e39b9a40374ee7`。Preflight 本身不产生性能结论。海光修复后 PMC 的行为沿用[海光验证报告](hygon_gems_profile_validation_20260923.md)。

## 收尾与边界

所有临时 KGS 与本地 SSH proxy 已停止，其他服务未改动。原始请求、响应、case 报告、scheduler 前后快照、厂商 profile 产物及 `hygon-evidence.tar.gz` 保存在 KG 本地 `runs/gems-preflight-20260923/`；远端现场保留，无删除实验数据。

完整 KGS `/evaluate` 尚需单独迁移 correctness 与 benchmark 的候选注入和报告契约；本次只把 Preflight 接回新 Gems 分支，不混入 Eval 功能开发。
