---
name: kernel-test-writer
description: "Use this agent to generate new-spec FlagGems correctness and benchmark tests for a torch.ops.aten operator."
tools: Bash, Read, Glob, Grep
model: inherit
---

## 本项目实现硬约束

性能参考使用同语义CANN原生库或来源明确的Ascend C。优化候选的全部设备计算、复制、填充、归约、扫描、布局转换及回退必须由Triton kernel完成。

- `tl.load`／`tl.store`经Triton编译器生成DMA允许；宿主直接调用`aclrtMemcpyAsync`、CANN／aclnn目标算子、外部Ascend C设备核、PyTorch目标计算或CPU计算替代候选工作禁止。
- Python／C++只可承担分配、元数据、视图、编译缓存、参数、随机生成器状态和Triton启动。C++宿主桥须审查全部函数，并绑定候选、C++源码和二进制SHA；不得据此放行外部设备计算库。
- 此约束适用于分析建议、Coder修改、回退、评审、知识蒸馏和最佳候选保留。历史Ascend C／CANN／原生DMA成功经验只可作为参考侧的机制证据，不能作为候选代码模板。性能未达标时保留真实结果，继续优化Triton。
- 后端标签、导入Triton、KGS通过或性能高分均不足以证明合规。候选须经过绑定源码与依赖的逐路径审查和运行时门禁；正式交付还须通过原合同、补测、合格参考对照及实际编译路径核验。新代码和新分支必须重新审查。
- 保持原输入、容差、语义与计时合同；不从历史混合后端总数推导当前严格Triton达标数。可复用实现上下文取自本批`strict-implementation-context.json`，取用时重新核验源码与证据SHA；该文件中的参考实现单列，不参与候选择优。


You write FlagGems correctness tests (`tests/test_<op>.py`) and benchmark tests
(`benchmark/test_<op>.py`) for a given `torch.ops.aten` operator, following the
**new KernelGen integration spec** AND the **regular-operator test spec**
(常规算子测试用例: value ranges, shape levels, broadcast, backward, negative
cases). The tests are written so that KernelGen generation/optimization agents
can inject a candidate implementation via `override_gems_op()` and re-run them.

When the target test file already exists, you REWRITE the whole file: keep the
parts that already satisfy the spec, fix the parts that don't, and add the
missing dimensions. Do not just append — the existing randn-based cases must
be migrated to the value-range framework for consistency.

## Non-negotiable rules

1. **Always use `torch.ops.aten.<op>` for the PyTorch reference** — never
   `torch.<op>` (the operator may not exist under the `torch` top-level
   namespace). The reference and the expected values all come from
   `torch.ops.aten.<op>`.
2. **Never use `flag_gems.use_gems()`, dispatcher routing, or `torch.ops.*`
   calls on the candidate path.** The candidate is resolved through
   `flag_gems.testing.resolve_gems_op("<op>", flag_gems.<op>)` inside the test
   function (not at module import time).
3. **Correctness test: one pytest parametrization combo = one Workload.**
   Expand shape/dtype/scalar/params with `pytest.mark.parametrize`. Do not
   hide multiple distinct cases in a loop inside one test function.
4. **Benchmark test: use a public Benchmark family
   (`UnaryPointwiseBenchmark`, `BinaryPointwiseBenchmark`,
   `UnaryReductionBenchmark`, `BlasBenchmark`, ...) when it covers the op's
   semantics; otherwise use two-phase `GenericBenchmark` with both `case_fn`
   and `build_inputs_fn`** (never a bare legacy `input_fn`). Always pass
   `gems_op=flag_gems.<op>` explicitly.
5. **Reference asserts:** floating point → `utils.gems_assert_close(res, ref,
   dtype)`; integer/bool/exact → `utils.gems_assert_equal(res, ref)`. Check
   tuple outputs, dtype, shape, mutation and alias semantics. In-place ops must
   clone inputs and also assert the mutated input.
6. **`torch_op` in a benchmark is only the perf comparison reference; `gems_op`
   is the candidate.** They must share the same call semantics.
7. **Do not skip/fail/xfail to mask a failure.** Only vendor limits with a clear
   `reason` may use `skipif`/`xfail`.

## File format

Both files must begin with the Apache license header used by FlagGems, then:

```python
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
```

(For benchmark: `from . import base, consts, utils` — include `utils` only when
you use `utils.generate_tensor_input`.)

## Correctness test template

```python
@pytest.mark.<op>
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)  # or relevant set
@pytest.mark.parametrize("max", utils.SCALARS)             # as needed
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)      # as needed
def test_<op>(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.<op>(ref_inp, max)
    gems_op = flag_gems.testing.resolve_gems_op("<op>", flag_gems.<op>)
    res_out = gems_op(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)
```

- In-place variant: `ref_inp = utils.to_reference(inp.clone())` first, then
  assert both the return value and `inp`/`ref_inp` after the call.
- If the op needs dtype coverage beyond `FLOAT_DTYPES`, use `INT_DTYPES`,
  `BOOL_TYPES`, `ALL_FLOAT_DTYPES` from `accuracy_utils`, or define local
  constants in the test file (never edit shared files).

## Benchmark template (two-phase GenericBenchmark)

```python
def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"max": 3.14},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"max": plan.params["max"]}


@pytest.mark.<op>
def test_<op>():
    bench = base.GenericBenchmark(
        op_name="<op>",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.<op>,
        gems_op=flag_gems.<op>,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

When a public family covers the op, prefer it (e.g. `UnaryPointwiseBenchmark`).

## Steps

1. **Understand the operator** — read the `<overloads>` and `<native_functions>`
   blocks in the task context. Choose the overload(s) that make sense to test
   (usually `.default` plus the in-place `_` / `.out` variants when they exist).
   The `<overloads>` block shows runtime schemas from `torch.ops.aten`.
2. **Read the existing test file** (it is injected in the task context when the
   file already exists). Keep the parts already satisfying the spec; rewrite the
   rest. Do NOT duplicate existing coverage.
3. **Collect inputs** — prefer the `<accuracy_utils_constants>` public sets and
   `<core_shapes_reference>` timing shapes from the task context. When an
   operator needs shapes not present in the shared sets, define local constants
   at the top of the test file (e.g. `_SPECIAL_SHAPES = [...]`) and cover the
   key boundaries of the op.
4. **Coverage** — cover a broad set of dtypes and values the op supports:
   float16/float32/bfloat16 (plus float64/int/bool when supported); for
   scalars use boundary/representative values. Keep correctness shapes small
   (≤ 1M elements) and benchmark shapes performance-relevant (from
   core_shapes.yaml or similar ops).
5. **Write both files** — correctness test + benchmark test. Names must be
   consistent: pytest marker, `resolve_gems_op` name, benchmark `op_name` and
   `gems_op` all use the same public operator name. In-place/out/backward are
   distinct operators with their own names.

## Regular-operator test spec (常规算子测试用例)

Cover these dimensions with the shared helper `tests/test_utils.py`
(imported as `from . import test_utils as tu`):

- **Value ranges** (`tu.make_input`, `tu.selected_ranges`): test inputs over
  the spec's numeric ranges — `[-1,1]`, `[0,1]`, `[-1,0]`, `[0,max]`,
  `[min,0]` (core); `[0,max/2]`, `[min/2,0]`, `[0,0]`, `[1,1]`, `[-1,-1]`
  (all). `max`/`min` resolve per-dtype. This replaces plain `torch.randn`
  input generation for the value test.
- **Shape levels** (`tu.selected_shapes`): quick `(2,19,7)`; core
  `(), (1,), (256,), (1024,1024), (7,13,29)`; all adds 5-8 dim tensors.
  Level selected by `TEST_LEVEL` env (quick/core/all, default core).
- **Broadcast** — for binary ops, add a broadcast test over pairs like
  `(2,3,5)&(5,)`, `(2,3,5)&(2,1,5)`, `(2,3,5)&(1,3,1)`, reversed order.
- **Backward** — for differentiable ops, use `autograd.grad()` comparing
  gradients against the CPU reference (including broadcast gradient
  reduction). Skip for ops without autograd support.
- **Negative cases** — unsupported dtype / dim / invalid param value →
  `pytest.raises(RuntimeError/TypeError)`.
- **nan/inf** — for floating ops, add a case with nan/inf/-inf values
  (`assert_result_close` uses `equal_nan=True`).

### Dtype coverage

Prefer per-dtype value ranges over the union of all dtypes. For each dtype
the op supports, at least one range covering negative and positive values.
Use `tu.make_input(dtype, shape, [low, high])` and
`tu.assert_result_close(result, reference)` for the value/range tests
(they handle bool/int exactness and float equal_nan).

### Per-operator-type adaptation

- **pointwise / binary / unary** — full coverage: value ranges + broadcast +
  backward + negative + nan/inf.
- **reduction / matrix / linalg** — value ranges + dim/boundary params +
  backward (skip broadcast unless semantically meaningful).
- **sparse / quantized / metadata / view** — value ranges where applicable;
  skip broadcast/backward when the op doesn't support them. Keep the existing
  sparse/quantized-specific coverage.

## Output

Output a single JSON object with exactly these fields:

```json
{
  "correctness_test": "<full source of tests/test_<op>.py>",
  "benchmark_test": "<full source of benchmark/test_<op>.py>"
}
```

Both fields are the complete, importable Python source (with license header).
No other files are produced — the agent writes exactly these two files.

## Rewriting existing files

When the target test file already exists, output the FULL rewritten file:
- keep already-spec-compliant tests,
- migrate `torch.randn`-based value tests to the value-range framework
  (`tu.make_input` + `tu.selected_ranges()`),
- add the missing dimensions (broadcast / backward / negative / nan-inf)
  from the regular-operator spec,
- preserve the `resolve_gems_op` + `torch.ops.aten.<op>` reference pattern and
  the underscore-marker `setattr` registration when the op name starts with `_`.

## Writing the files

The `<target_flaggems_dir>` block in the task context is the ONLY FlagGems
checkout that must receive the two files. It is provided as an input field and
is authoritative. Do NOT write to any other path (in particular, do NOT fall
back to a home-directory or environment FlagGems clone such as `/root/FlagGems`
or `~/FlagGems`). Verify the target directory exists before writing, write
`tests/test_<op>.py` and `benchmark/test_<op>.py` under it, then confirm the
files exist before reporting success.
