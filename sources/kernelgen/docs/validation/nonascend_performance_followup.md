# 非昇腾性能下降续查：dispatch 与其余 25 条

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

归档范围：本文及配套台账冻结于 2026-09-08 早期 Excel 的 44 条厂商/算子样本，当时海光复测数据尚未补齐。后续更新后的 62 项 pytest review 和其中 35 项重新生成、原生复测属于独立批次，不回填本历史统计；文中的“当前”均指本次历史审计时点。原始 Excel、kernel_todo_v2/ 和 runs/ 为本地实验归档，不随 Git 分发。

**除 dispatch 外，证据最强的两类原因是 workload 参数分布不同，以及旧 best 的 reference 计时异常偏高。目前不能把原表的大批降速归为候选或编译器整体变慢。** 本文承接[44 条原表失败主线](nonascend_retest_mainline.md)，只调查其中加速比低于 0.8 的 26 条；将已讨论的 PPU cholesky_solve_helper 单独记录后，其余为 25 条。数量单位均为厂商/算子记录，保留同时精度失败或资源超限的行。

本次仅分析归档，没有新增设备实测。旧 adapter、实习生原表、审计固定 Gems `34bd6d68928c8b0039c42987840929da52aa6a62` 的原生 `--level core` 是三组独立证据。后者通过不表示原表失败已关闭；交集只核对候选 SHA、记录中的 dtype/shape_detail/标量参数，未证明输入内容、stride、环境和调用入口完全相同。

## 已讨论的 dispatch：记录结论与计时边界

PPU cholesky_solve_helper 的旧 adapter baseline 走 `torch.ops.aten._cholesky_solve_helper`，candidate 通过显式 `gems_op` 直接调用。当前原生 benchmark 没有显式 `gems_op`，baseline 和注册后的 candidate 都经同一 ATen 入口；`use_gems` 上下文建立在 `get_latency` 外。其他 Gems benchmark 可能仍指定直接 `gems_op`，不能按“原生”两个字推断全部入口一致。

旧 headline 为 1.6878×，原表 0.4011×，当前原生 0.4167×。5 个匹配点的 reference 新/旧耗时几何比为 0.963，candidate 为 3.899。另有同设备、同输入的辅助直接调用/dispatch 对照，后者耗时为前者的 2.96–3.17 倍；该实验用 device events 包围 host loop，不能精确分解正式 benchmark 的全部差值，也不能当作纯 CPU dispatch 耗时。

标准 Triton `do_bench` 在 device start/end event 之间调用 `fn()`，测得的是设备事件间隔，不是 CPU 墙钟时间，也不是纯 kernel 活跃时间之和。CPU dispatch 若使设备等待后续提交，会间接影响结果；若这段工作被已有设备队列覆盖，则不一定完整反映。因此不能写成固定的“kernel 时间 + 全部 dispatch 时间”。厂商 FlagTree 的具体计时实现仍以目标镜像源码为准。[Triton 官方实现](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py)

公平比较的约定：应用替换性能使用相同 ATen 入口，核实 baseline 未被候选覆盖、candidate 确实注入；独立直接调用结果可用于诊断，但单独标明。固定测试 level 之外，还必须固定计时 mode；`operator` 的同步 host 计时不能与 `kernel` 的 device event 计时混算。这里只记录设计，不修改计时默认值。

证据：归档 `remote/pingtouge/cholesky-dispatch-timing/artifacts/dispatch-timing.json`，以及 Gems adapter `d64794e63b502cb836bc015a92a62c42de4be05a` / native `34bd6d68928c8b0039c42987840929da52aa6a62` 的对应 benchmark 与 `benchmark/base.py`。

## 1. 参数集合改变，暴露以前未计时的慢路径

**PPU index_select_backward：1 条，责任先分给 KGS adapter 参数计划和 Gems benchmark 维护方；有效慢路径的优化再交候选维护方。** 两次都是 15 个性能点，但 adapter 的 core 参数计划全部取 dim=0，原生随机选 dim。当前原生只有 5 点与旧记录完整签名匹配。

| 对比集合 | 旧 adapter | 当前原生 |
|---|---:|---:|
| 各自完整的 15 点 | 3.3911× | 0.6903× |
| 真正匹配的 5 点 | 3.1167× | 2.9961× |
| 原生其他 dim 的 10 点 | 无对应计时 | 0.3314× |

匹配 5 点的 candidate 耗时仅增加约 6.7%，reference 增加约 2.6%，加速比下降约 3.9%；完整集合的明显下降主要体现在未匹配的 10 点。两个分组的 shape/dtype 构成也不同，尚不能把组均值解释成只改变 dim 的实验。当前 0.6903× 接近原表 0.6266×，但原表输入缺失，仍属于部分解释。

**天数 expand_copy：1 条，先由测试维护方恢复自建 workload。** 原表明确使用自建 workload，得到 0.214×；旧为 0.939×，当前 core 为 0.9280×。检查旧计划和原生 lambda，两者仅在最后一维为 1 时扩成 2；保存的 15 个 core 性能点均未触发这一条件，主要测同形状复制。候选为此设置了复制快路径；真正扩展走通用路径，需要构造并传输两个元数据张量，并在 kernel 中逐维计算索引。原表若覆盖真实扩展，确实可能测到此前没优化的路径，但缺自建输入，不能把 0.214× 已定责于这段代码。此处没有增加算子专用门禁。

**天数 addbmm、addmm_、baddbmm_：3 条，先对齐测试范围，资源超限单独处理。** 原表记录分别有 242、242、230 个已完成性能点，而旧/当前 core 各有 15 点；原表还报告后续大 workload 超共享内存。不能将部分完成的几何平均与完整 core headline 直接比较，也不能删掉失败尾部后算通过。共享内存超限是否属于 candidate 仍需执行对象证据；合法配置超资源一般由实现维护方调整，并非看到编译失败就定责编译器。

以上 5 条（25 条的 20%）存在明确的参数计划或比较集合差异线索；不是 5 条根因均已闭环。没有实习生的命令和 checkout，不能进一步声称这些都由 Gems 新增 workload 或默认 comprehensive 模式导致。

## 2. 旧 reference 偶发慢值进入 best，旧加速比被抬高

**摩尔 2 条，占 25 条的 8%。责任是 KG best 选择/最终确认及 KGS 计时证据，不是要求 Coder 把内核再加速同样倍数。**

| 算子 | 旧 best | 当前原生 | 匹配点 reference 新/旧 | candidate 新/旧 |
|---|---:|---:|---:|---:|
| adaptive_avg_pool2d_backward | 11.8763× | 5.4932× | 0.447 | 0.967 |
| matmul_bias_activation | 1.7629× | 1.5415× | 0.779 | 0.978 |

adaptive_avg_pool2d_backward 的旧 best 有一个小输入 reference=11.24132 ms，其他 14 轮同签名中位数仅 0.00930 ms，约差 1209 倍；candidate 旧/当前分别 0.00536/0.00508 ms。旧 agent 已在 conclusion 指出异常，最终 ledger 仍选该轮为 best。这里确认的是异常没有退出 best 评选；底层原因缺原始事件和同期设备证据，尚不能在冷启动、设备干扰、缓存或 timer 问题之间定责。

matmul_bias_activation 也有旧 reference 为其他轮中位数约 8.7 倍的点。这解释旧值虚高的一部分；不能解释原表的全部 0.0619× / 0.3677×，更不能替代后者的精度失败归因。上述对比只有前者完整匹配 9/9 点，后者匹配 9/15 点。

处理设计：最终 best 独立复验，保存 reference/candidate 双边样本；reference 异常漂移时不凭单个高加速比接受 best。保留异常证据，不用其他轮中位数替换后冒充实测。该项属于计时与验收，pytest review 和源码 preflight 不能单独解决。

## 3. 不能漏掉 baseline 被换成候选的情况

**PPU scaled_dot_product_efficient_attention：当前审计有 1 条基线身份错误，占 4%；尚不是原表低性能的已确认原因。** benchmark 的 `torch_op` 实际绑定 Gems 函数，源码替换同时改变了 baseline；当前约 1× 不能解释成 Torch 突然变快。该行虽有 12 点时延交集，也从可用双边比较中排除。

责任在我方原生复测注入和 baseline 角色校验。pytest review 要分别核对 reference/candidate 的实际 callable，不能只检查候选总调用数大于零。Coder 禁止修改 Gems/Torch 的约束无法单独防住评测器自己替错 baseline。

## 4. 新恢复的证据与仍未解释的下降

从旧 `runs/flaggems-adapter-version/` 找回 8 份历史 ledger，逐份验证 best round 源码 SHA 与交付候选相同、旧加速比吻合：天数 mvlgamma_、addmm_、unbind_copy；摩尔 rnn_relu、unbind_copy、var；PPU unbind_copy；沐曦 unbind_copy。不能仅凭目录名推断它们使用了哪个 Gems revision。

其中 mvlgamma_ 新增 15/15 个完整匹配点，reference/candidate 新旧几何比为 0.999/0.994；addmm_ 新增 9/15 个匹配点，两边分别变为 0.876/0.879，加速比基本不变。其余 6 份补齐了旧证据，但当前审计没有有效性能点，仍不能逐点配对。这 8 份旧 best 没有发现满足“其他轮至少 3 次、reference 高于中位数 5 倍”的新异常点；该筛查不能证明全部计时稳定。

25 条目前按证据可互斥分成：

| 证据情况 | 数量 | 占比 |
|---|---:|---:|
| 完整签名匹配，双边耗时几何比均在 ±6% 内 | 9 | 36% |
| 部分签名匹配，交集加速比基本不变 | 3 | 12% |
| 旧 reference 异常有证据 | 2 | 8% |
| dim 计划不同，交集与完整集合差异显著 | 1 | 4% |
| 当前审计 baseline 身份错误，排除计时比较 | 1 | 4% |
| 无可用签名交集或无当前计时 | 9 | 36% |

这不是根因占比。前四行合计 15 条可以描述双边时延变化，仍不等于 15 条数值正确、环境一致或原表失败闭环。完整匹配的 9 条包括同时存在原表准确性/资源问题的行，不能用时延接近撤销这些失败。

摩尔 LSTM backward 和 masked index put 最值得追原始复测日志：旧 1.854× / 4.574×，原表 0.0039× / 0.0047×，当前同签名 core 为 1.8511× / 4.3958×。现有双边时延没有百倍退化；需要核对原表的实际输入、执行对象、计时单位与 mode，不能猜测填错列或直接归因编译器。

## 数据与复算

[完整 25 条证据 JSON](../operations/data/nonascend_performance_followup_20260908.json)保留三组加速比、逐点双边时延、恢复的 ledger 路径、候选 SHA 和 dispatch 单独记录。输入位于不随 Git 分发的授权实验归档，恢复后运行：

```bash
python3 scripts/analysis/analyze_nonascend_performance.py \
  --mainline docs/operations/data/nonascend_retest_mainline_20260908.json \
  --audit-root /data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01 \
  --runs-root /data/akg_kernel_bench_lite/kernelgen/runs \
  --output /tmp/nonascend_performance_followup.json
```

归档根目录下的 `performance-attribution/reference-spike-origin-review.json`、`baseline-binding-repro.json` 和 `index-select-workload-mismatch.json` 分别提供逐轮异常、baseline 绑定复现和 dim 分组证据。源码副作用等独立发现继续保留在原审查台账，不替代这里的性能归因。
