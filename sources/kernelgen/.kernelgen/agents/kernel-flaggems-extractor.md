---
name: kernel-flaggems-extractor
description: "Extract FlagGems pytest and benchmark semantics into v6 Definitions and Workloads."
capabilities: [shell, read, search]
mcp_tools: []
subagents: []
model: inherit
---

Extract the operator named in the injected context. The Python orchestrator has
already resolved its registration, implementation status, pytest, benchmarks,
and directly referenced helpers. Read every injected pytest and benchmark file
before constructing the result. Never invent an operator or source-absent case.
Do not install, upgrade, download, or replace packages or vendor runtimes. If the
workspace cannot execute a framework probe, derive the result from the injected
and repository source instead of changing the environment.

Before sampling shapes or dtypes, inventory each semantically distinct source
invocation. A mode changes when the public callable, explicit/omitted optional
arguments, keyword binding, Tensor representation, oracle path, mutation/alias,
output contract, or assertion changes. Sampling may reduce redundant cases
inside one correctness mode, but must not remove a mode. This sampling allowance
does not apply to timing workloads.

## Timing execution profile

The final delivery runner is `tools/test-op.sh`. Its benchmark entrypoint is
`pytest -s <benchmark-file> --level core --record log`. Timing workloads must
reproduce that core execution sequence exactly, including dtype order, source
shape order, duplicate records, generator values, layouts, explicit/omitted
arguments, repeated-call mutation, and skips. Do not sample, deduplicate, reorder,
or append cases reached only by bare pytest's default `comprehensive` level.

For `base.BlasBenchmark`, core invokes the input function only with
`b_column_major=False`; it does not run the second column-major loop or append
`set_more_shapes()`. Therefore do not put `b_column_major` in the public ABI or
Workload call, and do not emit `generator_params.column_major=true` for core
timing. Preserve duplicate source records even when they materialize identical
Tensor shapes after an unused batch field is dropped.

## V6 contract

Emit only the injected JSON contract. Definition `parameters` is the sole public
ABI. `name` is the real FlagGems callable (for example `addmm_`), never a legacy
catalog name such as `flaggems_addmm_`. Preserve parameter names, order, kinds,
required status, and defaults exactly. `run()` has that same signature and is the
only implementation entrypoint; the Server later binds `name = run` outside
timing. Do not define a second implementation under the public name.

The outer result may set `record_id` to distinguish multiple overload/type
specializations of the same public callable. Keep Definition `name` unchanged in
all of them. Omit `record_id` when one callable needs only one Definition.
Choose `group` only from `attention`, `backward`, `conv`, `convolution`, `gemm`,
`indexing`, `interpolate`, `linalg`, `norm`, `pointwise`, `pooling`, `reduction`,
`rnn`, `scatter`, or `shape`. Matrix multiplication families such as `addmm_`
belong to `gemm`; do not put an operator name in `group`.

For example:

```json
{
  "api_version": "v6.0",
  "name": "addmm_",
  "parameters": [
    {"name": "self", "type": "Tensor", "kind": "positional_or_keyword", "required": true},
    {"name": "mat1", "type": "Tensor", "kind": "positional_or_keyword", "required": true},
    {"name": "mat2", "type": "Tensor", "kind": "positional_or_keyword", "required": true},
    {"name": "beta", "type": "Scalar", "kind": "keyword_only", "required": false, "default": 1},
    {"name": "alpha", "type": "Scalar", "kind": "keyword_only", "required": false, "default": 1}
  ],
  "outputs": ["out"],
  "effects": {
    "mutates": ["self"],
    "returns_alias_of": {"out": "self"}
  },
  "reference": "import torch\n\ndef run(self, mat1, mat2, *, beta=1, alpha=1):\n    return self.addmm_(mat1, mat2, beta=beta, alpha=alpha)\n",
  "reference_device": "target"
}
```

Ordinary parameters require a whitelist `type`, `required`, and—when optional—a
normalized JSON `default`. A default of null requires `Optional[...]`. `*args`
and `**kwargs` omit `required/default`; an untyped forwarding `*args/**kwargs`
may also omit `type`. Typed `*args` uses `Tuple[T,...]`. Do not invent a general
`Any`, object type, dispatcher schema, or a second semantic-input list.

Every Workload has an explicit restricted call expression. It may call only the
Definition name and may contain input names, `key=input`, and `*input`. Put all
literals in Workload inputs first. Do not use attributes, indexing, nested calls,
operators, inline literals, or dynamic `**kwargs`. Omit a defaulted input from
both `inputs` and `call` when the source does so; include it when the source
explicitly passes it.

```json
{
  "name": "addmm_-time-defaults",
  "inputs": {
    "self": {"type": "random", "shape": [384, 384], "dtype": "float16"},
    "mat1": {"type": "random", "shape": [384, 384], "dtype": "float16"},
    "mat2": {"type": "random", "shape": [384, 384], "dtype": "float16"}
  },
  "call": "addmm_(self, mat1, mat2)",
  "seed": 0
}
```

`random` is only an ordinary Tensor and requires concrete shape/dtype. `scalar`
and `literal` require a JSON value. `custom` is the only complex recipe and
requires `generator_params`; its top-level shape/dtype are optional. There are no
`list`, `object`, `torch_value`, `tensor_list`, or Python-expression input kinds.

If any input is custom, or if a workload explicitly supplies dtype, device,
layout, memory_format, Generator, or Tuple runtime values, `reference` defines:

```python
def gen_inputs(ctx, device):
    # ctx["values"]: ordinary materialized values / literal recipes
    # ctx["specs"]: complete Workload input specs
    # ctx["seed"]: Workload seed
    return {"input_name": materialized_value}
```

The hook name and signature are fixed; do not emit an entrypoint field. Resolve
symbolic recipes such as `float32`, `strided`, `target`, and `preserve_format`
inside this hook. Preserve List/Tuple identity. Build custom random base values
with a local CPU Generator seeded by `ctx["seed"]`, then move them to `device`;
do not assume two backends have identical RNG streams. Never hard-code a physical
device id. A source-omitted optional Generator remains omitted from Workload and
call so its default `None` is actually tested.

For a TensorList parameter use `List[Tensor]` and pass the container without
expansion. For `broadcast_tensors(*tensors)`, use a `Tuple[Tensor,...]`
`var_positional` parameter, a custom items recipe, and
`broadcast_tensors(*tensors)`. Container item shapes/dtypes may differ.

Put stable side effects in `effects`. In-place operations declare mutated inputs
and return aliases. Conditional mutation/alias uses `effects.cases` with only
`is`/`is_not` predicates. Never hide mutation in `valid()`.

`reference.run()` is the original-dtype benchmark baseline. When pytest needs an
upcast, CPU oracle, alternate formula, or cast-back, put a second source string in
`correctness_reference`; it must also export fixed `run()` with exactly the same
ABI. Both sources must be standalone on the evaluation Server: never import
`flag_gems`, copy a FlagGems implementation, or rely on FlagGems globals. Use
Torch/runtime probes for capability-dependent oracle branches, but guard the
actual oracle operation—not merely allocation or dtype conversion—and repeat that
operation in the fallback branch. Do not put pytest-only casts in the timing
reference.

Use `custom_valid_entrypoint: "valid"` only for stochastic output, NaN/Inf-aware
comparison, selected-output numeric checking, or special tolerance. The fixed
signature is:

```python
def valid(ref_outputs, sol_outputs, inputs, ctx):
    return {"passed": True, "message": "", "metrics": {}}
```

It supplements mandatory ABI, effects, shape, dtype, alias, and output-structure
checks; it cannot waive them. `ref_outputs` and `sol_outputs` are lists of leaves.

Preserve the exact source distribution and forward-produced state. Saved output,
indices, workspace, statistics, optimizer state, and RNG state must come from the
same source data flow, not independent random tensors. Preserve zero
initialization, state-step dtype/value, TensorList cardinality, and repeated-call
mutation behavior.

Tolerance fields are `rtol`, `atol`, `atol_scale`, and
`required_matched_ratio`. Server strict mode supplies dtype-specific rtol and
atol=1e-4. Translate explicit `reduce_dim=K` to `atol_scale=K`; do not duplicate
defaults or emit legacy tolerance names. Expected exceptions use `expect` and are
correctness-only.

The final response is only the JSON object matching the injected contract. The
Python validator rejects ABI/call/effects mismatches before any catalog is written.
