# OperatorOptimize：优化模式与代码输入

`OperatorOptimizeWorkflow` 消费本地 Native Catalog 或 KGS 已安装的 Catalog，不再次抽取。Native 和 Gems adapter 均支持 SimpleOpt 和 KernelGen；Gems 的正确性、计时及精确用例重放始终来自原 pytest，不生成另一份 Native workload。来源选择与能力要求见 [Catalog 输入来源](catalog_input_sources.md)。不同芯片分别使用独立 workspace 和 KGS；复用抽取结果时传入同一份本地 Catalog。内容、目标环境和代码输入冻结后，续跑不能静默更换。

## 最小输入

```json
{
  "catalog_path": "/path/to/accepted-catalog",
  "operator": "scaled_dot_product_cudnn_attention",
  "optimization": {
    "definition_name": "scaled_dot_product_cudnn_attention",
    "eval_server_url": "http://127.0.0.1:23107",
    "mode": "kernelgen",
    "n_parallel": 2,
    "n_epoch": 2,
    "max_round": 2,
    "seed_code_path": "/path/to/previous-triton.py"
  }
}
```

```bash
kg run --definition scaled_dot_product_cudnn_attention \
  --catalog-path /path/to/accepted-catalog \
  --workspace ./campaign/operators/my_operator/ppu \
  --eval-server http://127.0.0.1:23107 \
  --n-parallel 2 --n-epoch 2 --max-round 2 \
  --seed-code-path /path/to/previous-triton.py
```

URL 是本地 SSH stdio proxy 的地址；端口仅为示例，以实际 `kg server` 实例为准。OperatorOptimize 是 `kg run` 的统一入口，支持 YAML Batch；独立 Catalog Python launcher 已删除。上面的 JSON 是对应的 Python Workflow 输入，不需要为 CLI 另存 input 文件。

`optimization.mode` 默认 `kernelgen`，复用 `KernelGenWorkflow`，`n_parallel` 和 `n_epoch` 默认均为 1，`max_round` 默认 10，Profile 默认启用。显式选择 `simple_opt` 时复用单 Coder 优化执行层，占一个本地 Coder lease；KernelGen 占 `n_parallel` 个 lease。资源池容量必须足够，不能把它等同于 KGS 请求线程或设备 slot。轮数、计时、超时和 Profile 配置传入原优化器。KernelGen 的所有 Coder 使用同一个上传 Bundle 的冻结评测快照，不回退到内置 Gems Catalog。

## 三种不同的代码职责

- Native Catalog 的 correctness reference 和 timing baseline 是评测契约，决定正确性和加速比，不由下面两个提示输入替换。
- `reference_triton_path`（可搭配 `reference_triton_prompt_path`）是只读设计参考，不承诺正确，不自动成为候选或 baseline。
- `seed_code_path` 是初始候选上下文。提交时只读取并冻结摘要，不执行、预验证或继承历史成绩；在新目标上仍经过原优化器的 Preflight、Eval 和最终确认。KernelGen 将其传给首个 epoch 的 Coders，后续 epoch 沿用正常选择逻辑。

reference 与 seed 可以同时提供，也可以均省略。任何文件变化都会使原计划无法续跑，必须保留原文件或创建新计划。已有 Catalog 的审核、Bundle 上传和目标 readiness 不跳过，审核通过不等于目标 reference 可执行。

## 结果、恢复与验证边界

SimpleOpt 保留原单 Coder ledger；KernelGen 保留 `stages/optimize/work/<epoch>R/agent*/` 的独立 ledger 和最终复测证据，不制造一个扁平汇总 ledger。下游 code review 使用与 KernelGen 输出匹配且已确认的获胜 Coder 文件。顶层 `kernelgen_output.json` 只是派生摘要，没有确认 winner 时不能报告优化成功。

Pipeline 续跑仍复用 `stages/optimize/work`，已完成的其他调用按凭据跳过。KernelGen 通过原 Coder ledger 恢复，不在外层增加 epoch checkpoint 或第二套轮次计数；当前适配层会重新进入 epoch 编排，尚不承诺跳过 Analyzer/Synthesis。取消在原模型安全点生效，正常、失败和取消都释放 Coder lease。

host 测试覆盖模式校验、Bundle/参数映射、摘要变化、两种权重与异常释放；真实设备验收单独记录，不能将模拟测试视为芯片端到端通过。
