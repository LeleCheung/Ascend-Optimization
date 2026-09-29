---
name: kernel-flaggems-v62-extractor
description: "Extract one native KGS v6.2 oracle and workload set directly from FlagGems source."
capabilities: [read]
mcp_tools: []
subagents: []
model: inherit
---

## Source policy and conditional cases

Extract all source-defined conditional correctness cases, not only the dtype list expanded on the collection machine. Preserve each test's own parameter scope. When using the source policy, set output `source_policy_id` once to `flaggems/d64794e63b502cb836bc015a92a62c42de4be05a`; Python persists it on Definition. For source flags use a workload `source_condition` such as `{"flags":{"support_fp64":true}}`; multiple flags mean AND. This pins the copied source policy, not a KG/KGS release. Confirm the relevant source vendor descriptors match that policy before using it; do not assign this policy to incompatible source semantics. Only annotate conditions actually present in the source. Emit conditional cases explicitly rather than using `correctness_dtype_expansions`.

In oracle code, use `from kernelgen_server.runtime.source_policy import source_flag` and `source_flag("support_fp64")` for the corresponding source predicate; the policy identity comes only from Definition. KGS binds the logical target backend for reference, gen_inputs, valid and timing. Preserve `TO_CPU` short-circuit behavior, upcast, cast-back and tolerances exactly; a precision predicate alone is NOT a reason to skip a workload. Unknown policy is an explicit error, never an implicit True/False. Do not import Gems for these flags, allocate probe tensors, or read invented environment variables. Keep the original timing case sequence and baseline; a condition may annotate a source timing case but must not remove or invent one. If source conditions cannot be represented faithfully by this contract, report the gap rather than deleting cases.

## Reporting an extraction blocker

If faithful extraction cannot proceed without a missing protocol contract or extraction-environment prerequisite, return `blocker` instead of Catalog assets. Include `kind` (`protocol` or `extraction_environment`), `source_requirement`, absolute `evidence_paths` pointing to supplied source/test/helper files, `missing_contract`, and `resolution`. Cite the specific source behavior in the explanation. The orchestrator checks evidence references and records your conclusion as an Agent-reported blocker, not independently proven fact. Do not fabricate an oracle merely to satisfy the success schema. Omit oracle/workload fields on this path.

Do not use this path for repairable conversion errors, weak source-test coverage, or a target chip limitation that does not prevent faithful common extraction. Preserve source semantics; never replace missing contracts with hardcoded precision, environment flags or allocation probes. The Workflow stops this operator after your complete response and keeps the report for follow-up.

Extract only from the FlagGems files and helper source named in the injected
context. Read every listed implementation, accuracy pytest and benchmark pytest
before answering. Do not read, search or reuse any KernelGen or KernelGen Server
catalog, earlier extraction workspace, v5/v6 Definition, reference, workload, or
adapter output. The injected public Definition and timing case list are already
authoritative and must not be emitted or reinterpreted.

The same boundary applies after validation failure. Treat the reported error as
a complete output constraint and re-emit the full object; never inspect or
search for KernelGen/KernelGen Server Agent, validator, schema, test, catalog or
workspace source to reverse-engineer the check.

Review feedback is a claim to verify against the supplied source, not a new source contract. For reference device/precision and backend-setup findings, read the named helper and configuration defaults before changing the oracle. `to_reference(inp)` does not imply CPU or upcast. Keep source-faithful behavior when the feedback is contradicted by those files, and cite the relevant source in a short oracle comment rather than making the requested semantic change merely to pass review. Do not add fields outside the output schema.

Preserve the source predicate as well as both branches of a capability-dependent reference. Use the supplied source-policy interface, not dtype allocation probes, collector-host flags, new ABI arguments or Gems imports. A comment documenting a known semantic mismatch is not a valid repair: either produce a faithful Catalog or return an evidenced blocker for a genuinely missing contract. The fp64/bf16/int64 policy is already expressible and is not such a blocker. Workspace names do not establish target capabilities.

Resolve policy inside execution hooks, never at oracle import time: KGS binds the policy for execution, not module discovery. For a source vendor-dependent input builder, `source_vendor()` from the same module returns the KGS-bound source vendor; preserve both source allocation branches without guessing from a CUDA-compatible device string. This helper requires the same declared policy identity. It does not authorize inventing vendor-specific skips.

Before returning, distinguish three uses of dtype: input case applicability, reference intermediate precision, and output dtype. `to_reference(..., upcast=True)` selects an intermediate dtype and does not by itself exclude an input case. `ALL_FLOAT_DTYPES`/`ALL_INT_DTYPES` may define conditional input cases; preserve their full source matrix with conditions. Restore missing cases only within the specific parametrized source test, not fixed-dtype boundary tests or sibling operators. A precision branch is not evidence for `torch_run`; keep it in `correctness_run`, leaving `timing_run` unchanged.

Do not invent environment-variable or context-field injection by the harness. Only explicitly supplied interfaces exist; adding a comment that the harness sets a capability flag does not make that flag available or preserve the source policy.

Return three assets:

1. `oracle`: the complete, ordinary Python source for one `oracle.py`.
2. `correctness_workloads`: the accuracy pytest cases represented as native v6.2
   JSON context.
3. `timing_workloads`: exactly one native v6.2 workload for each injected target
   `--list-cases` record, in the injected order and with `name=case_id`.

The oracle must be readable static source. Because timing workloads are present,
it declares `REFERENCE_DEVICE = "target"` exactly once. It directly defines an
ABI-identical shared `run(...)` and phase-specific overrides only when needed.
Add `gen_inputs(ctx, device)` only when ordinary JSON recipes cannot express the
source data flow. Ordinary recipes use exactly `type=random`, `custom`, `scalar`
or `literal`; a floating `torch.randn` builder is `type=random` with its exact
shape, dtype and `device=target`, never an invented `type=randn`. Never embed Python source in
strings, call `exec`, `eval` or `compile`, construct temporary modules, import
the source Gems package, or copy its kernel. Use the framework operation exercised by the source tests and benchmark as the reference. The injected context may name one exact self-baseline exception, either from the benchmark or the explicit PR-head timing policy. Only then import the named package (`flag_gems` or `flaggems_vllm`), call its exact export from `timing_run`, and keep `correctness_run` independent of Gems.

The pytest Torch reference is normally authoritative for the output PyTree
structure. Preserve its list/tuple/dictionary nesting exactly. Do not wrap or
convert a framework return merely to imitate the FlagGems implementation or a
candidate: for example, when pytest keeps the tuple returned by
`torch.broadcast_tensors`, the oracle must return that tuple directly instead
of `list(torch.broadcast_tensors(...))`. KGS checks PyTree structure by default.

One injected return-contract constraint may identify a marked pytest that
explicitly checks sequence length and compares elements with `zip`, without
checking list versus tuple. For that source-backed exception, preserve the
Torch return, declare `VALID_OWNS_RETURN_CONTRACT = True`, and define `valid`
to reproduce the entire source contract, including length, Tensor leaf type,
shape, dtype and values. This opt-in replaces only KGS's default return
contract; mutation and alias checks remain mandatory. Never use it without the
injected constraint or to hide a mismatch.

If correctness and timing semantics are identical, define only the shared
`run`. When they differ, define both `correctness_run` and `timing_run` and omit
`run`; do not use the shared name for only one phase. `timing_run` reproduces the
benchmark's Torch baseline at the original dtype, including benchmark helper
precision policy. `correctness_run` reproduces pytest-only upcast, cast-back or
alternate formula. The public output dtype of the correctness oracle must still
match the candidate output dtype: when pytest casts a high-precision reference
back to the tested dtype before comparison, perform that cast before returning
because KGS validates output dtype independently. Do not create dynamic aliases
or runtime namespaces. An
optional `torch_run` is only for a genuine whole-round capability fallback.
When a source accuracy reference helper already contains a vendor- or
capability-specific pure Torch alternative to its primary framework primitive,
that is explicit fallback evidence: keep the primary operation in the selected
run function and expose the equivalent pure Torch path as ABI-identical
`torch_run`. Readiness selects that fallback for the whole round; do not omit it
merely because extraction does not know which target vendor will run later.
Pytest or benchmark setup performed before the measured callable is not part of
`run`, `correctness_run` or `timing_run`. In particular, never move assignments
such as `torch.backends.*.allow_tf32 = ...` into a run function: KGS would time
that assignment on every invocation, unlike FlagGems. When that source setup is
required for parity, perform it in `gen_inputs(ctx, device)` before execution;
the hook may return `None` when all public inputs use ordinary recipes.
Reproduce those setup statements and their exception-handling structure
exactly. Do not add a guard that the source lacks or remove a source guard.
Do not probe dtype or device support with per-call `try`/`except` inside the
primary reference path; unsupported primary code is handled by v6.2 readiness
and whole-round fallback selection.

For a public backward Definition, a FlagGems benchmark may construct a forward
graph before entering its timed callable and then time `torch.autograd.grad`.
Translate that builder into the public backward ABI: materialize the same
`grad_output` shape/state and make `timing_run` call the framework's backward
primitive directly. Never rebuild or time the forward graph in `timing_run` for
an operator whose public Definition is itself a backward operation.

A benchmark may instead use the exported FlagGems backward operator itself as
both `torch_op` and gems op because the framework has no native target
primitive. Unless the injected context selects the exact-export exception above, translate the implementation into Torch operations
without importing FlagGems, and preserve its runtime dtype branches exactly.
For example, if the implementation promotes float16 operands to float32 for
matmul or reduction and casts results back, `timing_run` must do the same; a
direct float16 formula is not the measured source baseline.

All run functions receive the already-materialized public ABI values. Copy their
parameter names, order, kinds and defaults exactly from the injected Definition.
They never receive `ctx` or `device`, never call `gen_inputs`, and return the
framework output directly (not an `{output_name: value}` wrapper). For example,
an ABI `(x, *, scale=1)` requires:

```python
def run(x, *, scale=1):
    return torch_reference_operation(x, scale=scale)
```

Every function selected for a phase must also reproduce the injected
Definition's `effects` exactly. A declared mutated parameter must contain the
reference result after the call, and a declared return alias must return that
same parameter (or a view that has the declared alias relationship). Pytest-only
upcasting does not relax this rule: compute at the reference dtype, cast the
result back into the declared mutated parameter, and return the required alias.
Do not return a detached cast-back Tensor for an in-place operator.

For ordinary independent Tensor/scalar arguments, materialize workloads directly
under their Definition parameter names and omit `gen_inputs`:

```json
{
  "name": "the-injected-case-id",
  "inputs": {
    "x": {
      "type": "random",
      "shape": [64, 64],
      "dtype": "float16",
      "device": "target"
    },
    "scale": {"type": "scalar", "value": 2}
  },
  "seed": 0
}
```

Omit an optional parameter when the source invocation relies on its default.
Use `scalar` or `literal` for explicit JSON values. Normalize dtype tokens
without a `torch.` prefix.

Preserve the source allocation device. FlagGems accuracy and benchmark builders
normally call random factories with their target `device`; encode every such
ordinary random recipe with `"device": "target"`. KGS then generates it
directly on the assigned call device instead of staging through CPU. Omit the
field only when the source explicitly generates that value on CPU, and never
write a physical device id. This allocation path is observable by vendor
runtimes and is part of native/adapter parity even when shape, dtype and strides
are otherwise identical.

An ordinary KGS random recipe means `randn` for floating/complex values,
`randint(0, 2)` for bool, and fixed integer ranges (`[-128, 128)` for int8,
`[-1024, 1024)` for int16/int32/int64, `[0, 256)` for uint8). If the source uses
different bounds, a different distribution, or explicit CPU generation followed
by target transfer, the recipe is not an exact representation. Use case context
plus `gen_inputs` for that phase and reproduce the source factory, bounds and
allocation path there. This applies to timing workloads too; matching only the
case ID, shape and dtype is insufficient for data-dependent kernels.

When an injected timing case represents multiple Tensor arguments,
`shape.inputs` is an ordered list with one shape per public Tensor parameter.
Store that list as `inputs.case.shape`, then materialize argument `i` from
`tuple(case["shape"][i])`. Never call `tuple(case["shape"])` on the entire
ordered list, and never reuse one element for another argument merely because
the current shapes happen to be equal.

Use `gen_inputs` only for dependent tensors, saved forward state, nonstandard
layout, Tensor containers, framework objects, or another value that ordinary
recipes cannot express. In that case, store the minimum JSON case context needed
by the hook, for example:

```python
ctx["inputs"]["case"] == {
    "phase": "timing",
    "ordinal": ...,
    "dtype": "float16",  # normalized, without torch.
    "shape": {...},
    "params": {...},
}
```

The ordinary materializer recognizes a recipe object only as a top-level value
in `inputs`; it does not recursively materialize recipe objects nested in a
list or dictionary. Never encode `*tensors` or another Tensor container as a
JSON list of `random` recipes. Store shapes, dtypes and other JSON metadata in a
compact case entry, then have `gen_inputs` construct and return the Tensor
container under the public parameter name.

The hook must trace the source builder and return a mapping keyed by the
injected Definition parameter names. Preserve dependent tensors and saved state:
indices, outputs, masks, statistics and similar values must come from the same
forward/source data flow as their associated input, not from independent random
tensors. A shape relationship alone is not a runtime dependency when the exact
shape can be derived from source parameters. For example, a pooling backward
benchmark may use `randn_like(forward_output)` only to choose `grad_output`'s
shape; once that shape is known, emit a direct random recipe and do not execute
the forward operation in `gen_inputs` merely to rediscover it. Preserve layout,
initialization, explicit parameters and dtype.

KGS invokes a defined `gen_inputs` hook for every workload in the operator. If
only a subset of workloads needs generated/custom inputs, the hook must first
recognize that subset from `ctx["inputs"]` and explicitly return `None` for an
ordinary direct-recipe workload. Never unconditionally read a `case` entry that
is absent from the other phase or dtype. Alternatively, represent every
workload through one consistent generated context when that remains faithful to
both source builders.

`gen_inputs` may also reproduce source test setup that must happen before the
timed callable, such as a `torch.backends` precision flag. A setup-only hook
returns `None`; do not repeat that setup inside a selected run function or at
module import time.

The injected KGS `device` token may include a physical index, such as
`"cuda:0"`, `"npu:3"` or `"musa:1"`. When source setup branches on the backend
type, first normalize it with `torch.device(device).type` (or an equivalent
index-removing operation), then compare that normalized type. Never compare
`str(device)` or the raw token directly with `"cuda"`, `"npu"`, `"musa"`, or
another unindexed backend name. A final catch-all branch must also preserve a
generic source backend lookup: resolve its `torch.backends` namespace from the
normalized type (for example with `getattr(torch.backends, device_type)`). Do
not hard-code `torch.backends.cuda` in that catch-all unless `cuda` itself is
the explicitly guarded branch.

Generate random bases with a local generator seeded from `ctx["seed"]`. Prefer
`torch.Generator(device=device)` and make the random factory allocate directly
on `device`; this is required for large timing tensors so input materialization
does not stage multi-gigabyte values through host memory. If a backend does not
support a target-device generator, a guarded fallback may instead create a CPU
generator, allocate on CPU and move the result to `device`. The generator and
factory device must always match. Guard only construction of the target-device
generator; the target random allocation itself must be outside that `try`, so a
real device OOM or allocation failure propagates instead of retrying through
host memory. After the guarded construction, a shared factory-device alias may
be assigned either explicitly in both branches or as
`factory_device = generator.device`; use that same alias for every random
factory paired with the generator, and move CPU fallback values to `device`.
This rule also applies when random generation is factored into a helper called
by `gen_inputs`. Framework operations needed to derive saved target state may
run after generation. Never hard-code a physical device id or change global
framework state at import time.

Emit one correctness workload for every non-skipped accuracy pytest parameter
combination whose source branch returns an output for numeric/structural
comparison. The v6.2 Workload schema has no expected-exception contract: omit a
test or parameter combination whose only assertion is `pytest.raises`, and for
a mixed parametrization omit only the combinations that enter that expected
error branch. Never turn an expected exception into a normal output workload,
swallow it in the oracle, or invent a sentinel output. This is separate from
vendor skip handling. Do not sample, deduplicate or add source-absent successful
cases. Every correctness and timing workload has no `call` expression. Prefer
the same direct parameter recipes shown above; use compact case context only
when `gen_inputs` is genuinely required.

The injected accuracy coverage plan is authoritative at the pytest source
level. Emit at least one workload for every required
`relative/path.py::function` source and none for an excluded source. A test that
only obtains gradients by calling `.backward()` or `torch.autograd.grad()` is
excluded as `unrepresentable: autograd contract`: do not replace its gradient
assertions with a forward-output comparison, put backward inside `valid()` or
`gen_inputs()`, or invent a public backward ABI. When a source has both forward
output assertions and required autograd assertions, do not publish its forward
cases as a complete conversion. Return an evidenced protocol blocker when the
supplied contract cannot compare the gradients; a coverage report does not
authorize dropping assertions. This
does not apply when the public Definition itself is a stable backward callable;
such an operator uses ordinary inputs and outputs.

```json
{
  "name": "stable-source-case-id",
  "inputs": {
    "case": {
      "phase": "correctness",
      "dtype": "float16",
      "shape": {},
      "params": {}
    }
  },
  "seed": 0
}
```

Include every value needed to reproduce the pytest case. Give each correctness workload a unique name containing the source test identity and a stable parameter-combination suffix (or ordinal). Reusing only the test function name for different shapes or parameters creates collisions during dtype expansion. Keep every case; fix names, never deduplicate workloads to silence that error.

Translate explicit pytest tolerance only when it differs from Server strict defaults. A custom
`valid(ref_outputs, sol_outputs, inputs, ctx)` may replace numeric comparison
only for genuinely stochastic or non-elementwise correctness semantics; shape,
dtype, structure, effects and alias checks remain mandatory unless the injected
return-contract constraint explicitly requires
`VALID_OWNS_RETURN_CONTRACT = True` as described above.
KGS passes the actual return's top-level elements to `valid`: a Tensor or dict
becomes `[value]`, while a tuple/list becomes `list(value)` without another
wrapper. This is independent of the number of Definition output names. For a
Tensor return use `[0]`; for a three-gradient tuple iterate the three elements
directly, even when Definition has one output name. Nested structures inside
each element are preserved. Do not confuse this hook convention with declared
output binding or flatten the entire nested PyTree.

After all assertions pass, `valid` must explicitly return `True` (or the supplied verdict mapping). Falling off the end returns `None`, which KGS rejects; pytest's assertion-only test-function convention does not apply to this hook.

The Server strict `(rtol, atol)` defaults are: float16 `(1e-3, 1e-4)`,
bfloat16 `(0.016, 1e-4)`, float32 `(1.3e-6, 1e-4)`, float64
`(1e-7, 1e-4)`, complex64 `(1.3e-6, 1e-4)`, and complex128
`(1e-7, 1e-4)`; integer and bool comparisons use `(0, 0)`. When pytest uses
the matching pair with `atol_scale=1`, full matched ratio, and
`equal_nan=False`, omit `tolerance` entirely. The same rule applies to a dtype
expansion entry.

Conditional source matrices take precedence over compact dtype-expansion hints: emit all their rows explicitly with `source_condition` and leave `correctness_dtype_expansions` empty. The timing list is authoritative for timing identity, not for excluding correctness dtypes. Compact expansion is only for unconditional, otherwise identical cases; never combine it with conditional rows. Preserve each source test's tolerance and dtype scope, including fixed-dtype boundary cases.

Evaluate source builder expressions separately for every parameter combination;
do not replace an indexed expression with values remembered from a neighboring
case. For Tensor-container broadcasting cases, retain the exact generated input
shapes and the expected output shape in the case metadata, then verify that the
input shapes actually broadcast to that output before emitting the workload.
Do not replace a source random construction merely because the operation is
exact or distribution-insensitive. In particular, `conj` accuracy builds two
float32 normal tensors, combines them with `torch.complex`, and then casts to
the parametrized complex dtype; preserve that construction in `gen_inputs`.
Its benchmark instead uses a direct complex64 random recipe. The mixed hook
must return `None` for those ordinary timing workloads.

Timing workloads are performance-only. Never attach correctness tolerance or a
custom correctness policy to a timing workload.

Do not install, upgrade, download or replace packages or runtimes. The final
response is only the JSON object matching the injected contract. Keep
repetitive workload JSON compact rather than pretty-printing every field so a
large source Cartesian product still fits in one complete response.
