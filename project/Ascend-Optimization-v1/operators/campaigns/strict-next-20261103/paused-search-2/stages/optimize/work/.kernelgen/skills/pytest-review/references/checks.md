# 脚本使用与证据边界

两个脚本随 skill 分发，无需安装新依赖：源码扫描只用标准库，运行观测使用目标环境已有 pytest。使用当前 skill 目录的绝对路径，不能假定工作目录就是 skill 所在位置。运行观测会执行用户指定的测试，因此通过项目已有 KGS Debug Job 在被调度的目标设备执行；不在 Agent 本机推断厂商能力。脚本不启动 Server、不占用额外卡、不创建审核 agent。

## 默认：一次 reference-as-solution

在目标环境通过已有正式注入工具准备好 reference solution；探针不替换 Gems 源码，也不实现另一套注入器。`--candidate` 必须指向实际加载的 solution 文件，`--reference-source` 指向其来源文件；两者内容 hash 不同则在运行前返回 `NEEDS_FIX`，不会执行 pytest。此身份校验证明拷贝关系，不证明来源本身的数学正确性。

```bash
python3 /path/to/pytest-review/scripts/probe_pytest.py \
  --gems-root /path/to/FlagGems \
  --candidate /path/to/injected/reference_solution.py --entrypoint run \
  --reference-source /path/to/prepared/reference_solution.py \
  --phase reference-as-solution \
  --benchmark-file /path/to/FlagGems/benchmark/base.py \
  --output /path/to/artifacts/readiness.json \
  -- tests/test_operator.py benchmark/test_operator.py -m operator -q --level core
```

上述入口与测试路径须换成现场真实值，正式注入器的参数放在 `--` 后；没有可用注入器时先记录准备失败，不能直接运行未注入测试后宣布通过。只检查 correctness 时可不传 `--benchmark-file`，但这不构成 benchmark 入口已经验证的证据。默认不再要求先跑独立 reference、源码扫描或生成 `--expect` 后重复执行。

输出 schema 为 `pytest-review-readiness/v1`。退出码 0 对应 `READY`，1 对应 `NEEDS_FIX`；命令行配置错误退出 2，不能当作完成验证。`problems` 保留阻塞证据、可能的四类 P0 signal、责任及下一步；尚不能确定分类时 signal 为 null。`suggestions` 中的 P1 覆盖与语义未独立证明等提示不阻塞。保留旧 `findings`、`unverified`、`audit_status` 和原生 `exit_code` 作为证据，不能用旧审计状态覆盖新的准备状态。API/编译信号来自明确异常类型，其他失败保留待定位；不以 AssertionError 自动判定 reference 语义错误。

全部 skip、实际候选旁路、基线被候选覆盖、来源变更、观察器被替换或无法证明实际入口会阻塞。部分 skip 与未全面审核 workload 不阻塞。有限正计时或低加速比不是本阶段的性能验收条件；不完整调用角色导致注入无法确认时仍不能放行。正式计时关闭观测后单独运行。

混合运行原生 `tests/` 与 `benchmark/` 时，各已收集阶段必须实际执行，benchmark 通过不能掩盖 accuracy 全 skip；报告使用 `NO_EXECUTED_TESTS` 的 `detail.phase` 标明缺失阶段。各阶段均有执行时，部分 workload skip 仍是 P1 建议，不要求所有 case 都不跳过。

## 覆盖与低精度检查

对每个待审核算子建立覆盖摘要：来源（Gems 已有实现或 ATen 开发项）、开发单位/KT2 标记、`native_functions.yaml` dtype 声明、Gems 注册 dtype、CUDA 实测结果、各 phase 的 case 数及缺失维度。低精度集合固定检查 `int8`、`uint8`、`float8_e4m3`、`float8_e5m2`；只记录实际支持的类型，不把“不支持”转成失败 case。

case 少于 100 时，给出按算子约束补到至少 100 的建议，涵盖数值范围、shape、dtype、广播和参数边界；审核本身不修改 pytest。shape 有固定维度时遵循原生限制。原始 pytest 没有覆盖的边界域通常放入 `suggestions` 或人工 review 记录，只有产品或算子契约明确要求时才升级阻塞。覆盖检查不替换 reference-as-solution，也不允许通过删除 workload、放宽 oracle 或修改 skip 来达到数量要求。

## 可选：源码扫描

```bash
python3 /path/to/pytest-review/scripts/review_source.py \
  --candidate /path/to/candidate.py \
  --test /path/to/FlagGems/tests/test_operator.py \
  --test /path/to/FlagGems/benchmark/test_operator.py \
  --output /path/to/artifacts/source-review.json
```

返回 0 表示完成扫描，2 表示输入／语法错误。成功扫描的状态始终为 `REVIEW_REQUIRED`，不是候选通过。检查显式框架赋值、setattr、注册／全局设置、Gems baseline、直接 ATen 和按时间设 seed 等线索；动态别名、间接调用和数学语义需要人工阅读。每条线索给出文件、行号和 hash，不自动给责任或严重级别定论。

## 可选：原生 pytest 深入观测

以下是运行结构，测试路径、marker、candidate 路径和已有注入参数必须换成审核现场的真实值。脚本不会替用户配置 candidate path，也不会覆盖 Gems 源文件。先使用干净 reference checkout 运行 `--phase reference`，再在已有注入工具的环境中运行 `--phase candidate`。如需携带插件参数，放在 `--` 后与原生 pytest 参数一起传入。必须分别起新进程，不能在一个 Python 进程里连续跑两种 phase。

```bash
python3 /path/to/pytest-review/scripts/probe_pytest.py \
  --gems-root /path/to/FlagGems \
  --candidate /path/to/candidate.py --entrypoint run \
  --phase candidate \
  --benchmark-file /path/to/FlagGems/benchmark/base.py \
  --output /path/to/artifacts/candidate-benchmark-probe.json \
  -- benchmark/test_operator.py -m operator -q --level core
```

accuracy 使用相同命令结构，去掉 `--benchmark-file`，把 `--` 后换成原生 accuracy 命令。benchmark 自动补充 `--level core`，拒绝显式的非 core 参数。不支持 xdist／forked 或已有 profiler；观测运行只用单进程。主进程 Python 入口通过 candidate 文件路径及 entrypoint 名称识别，不按自报调用数。原生代码若使用 JIT-only 入口、线程／子进程、动态生成或同名嵌套函数，需要专门验证，不能认为这个探针具有完整覆盖能力。

第一次报告包含 `collected`、`identity` 和 `packages`。审核 collection 与 skip 后，制作对应 phase 的 expected JSON，包含这三个字段和 `allowed_skips`（精确 nodeid 列表），通过 `--expect /path/to/expected.json` 复查。不要不经审核直接把观察结果复制为预期：需要与原生 collection／case 清单核对。identity 覆盖 Gems root 下 Python 和常见配置文件（含未跟踪文件）、commit 和 candidate 文件 hash；外部 helper、插件、非配置数据和镜像／驱动身份仍须在审核记录单独绑定。报告输出必须在 Gems root 外，避免改变输入身份。另记录实际导入的 `flag_gems.__file__`；若指向另一个 checkout，则报告 `FRAMEWORK_IMPORT_MISMATCH`，不根据传入的路径假定运行了正确版本。

benchmark 角色根据原生 `base.py` 中 `metric.latency_base = self.get_latency(...)` 与 `metric.latency = self.get_latency(...)` 的实际调用点识别，不能靠函数名字或“第一次调用一定是基线”猜测。其他形式明确输出未知角色／缺失计时，不强适配。每个 metric 实例对应一个被观测 case，记录两种角色的 candidate 调用数、入口、有限正计时和可用的 Torch 输入元数据。该 case ordinal 仅在本次运行有效，不是跨版本输入一致性证明。

| 退出码／状态 | 意义 |
|---|---|
| 1 / ISSUES_FOUND | 发现实际旁路、baseline 调用候选、全 skip、collection 差异、pytest 错误、无效／不完整计时、可观察的输入污染风险或源文件变更。查看 findings 与原始 reports。 |
| 2 / REVIEW_REQUIRED | 已完成可支持的观测，仍有语义、输入独立性、身份／skip 审核或测试结构未验证。首版正常运行也会保留输入／reference 语义未确认项。 |
| 0 / CHECKS_PASSED | 仅表示所实现检查无问题且无未验证项；不等于算子或数学语义获批。首版没有消除语义未确认项的自动开关。 |

报告永远保留原生 pytest exit_code 和 setup/call/teardown 结果；probe 退出码与 pytest 退出码不能混用。probe 耗时包含观测开销，不计入 best 或加速比。正式计时仍由关闭观测的原生运行产生，本工具不修改其验收或选 best 逻辑。

源码 hash 和执行观测用于复查意外回归，不构成对任意恶意 Python 的安全隔离。候选可以在同权限进程中干扰观察器，前后完整性检查只能覆盖部分行为；首版报告不是 BLOCK 审核凭据或控制层放行令牌。

## 并行 Ascend profiler 输出

源码快照排除 Gems 根目录 `.flaggems_ascend_profile_*` 子目录中未被 Git 跟踪的 JSON 运行产物，避免其他并行任务的 trace/metadata 创建或清理造成源码变化误报。该规则不排除这些目录中的 Python、YAML 等源文件，也不排除 Git 已跟踪的 JSON 或其他目录中的 JSON 配置。candidate 和 reference 的独立 hash 校验保持不变。报告另存 `observer_intact`，源码变化与观察器被替换分别记录证据；旧报告缺少该字段时不能仅删除告警即作为新一次 READY 的证明。

并发任务删除上述临时目录时，源码遍历只容忍未跟踪 profiler 目录的 FileNotFoundError；源码目录、含已跟踪文件的目录以及其他读取错误仍使检查失败。曾因该竞态中断的报告必须重新执行，不能把日志中的 pytest passed 直接补写为 READY。
