"""Prompt policy and narrow source-backed exceptions for FlagGems extraction."""

REFERENCE_DTYPE_GUIDANCE = (
    "Workload tensor dtypes are candidate input dtypes. reference.run() stays at "
    "the benchmark's original dtype. If pytest uses a reference-only transform, "
    "put it in definition.correctness_reference, whose fixed run() must have the "
    "same v6 ABI (v6.2 uses oracle.py correctness_run). Original pytest precision "
    "is authoritative: float64/.double() is allowed when source-required. Preserve "
    "TO_CPU/support_fp64 branches; upcast=True is not necessarily float64. "
    "Do not substitute the candidate dtype or invent a capability probe/fallback "
    "to make the reference run. Preserve the candidate output dtype separately from the temporary oracle "
    "compute dtype. Omit correctness_reference when both paths are identical."
)

GENERAL_EXTRACTION_PRINCIPLES = """
Apply these rules to every operator, including operators with no few-shot rule:

1. Pytest is the correctness specification. Trace parametrization, fixtures,
   helpers, input construction, the exact public call, reference transforms, and
   assertions. Dispatcher schemas may disambiguate a source signature, but never
   replace the Python callable and tests as the public ABI source.
2. The benchmark input_fn and Benchmark subclass are an independent performance
   specification. Timing is the exact sequence executed by tools/test-op.sh:
   pytest -s <benchmark-file> --level core --record log. Preserve dtype order,
   source shape order, duplicate records, values, kwargs, layout/stride/view state,
   skips, and repeated-call mutation. Never sample, deduplicate, reorder, append
   comprehensive-only set_more_shapes/layout cases, or reuse correctness generation
   unless the source paths are actually identical. Input-generation controls belong
   in generator_params, never in parameters or call; in particular BlasBenchmark
   core uses b_column_major=False only.
3. Inventory every semantically distinct pytest invocation mode before sampling.
   A mode changes when public callable, optional/keyword values, tensor
   representation, oracle path, mutation/alias behavior, output contract, or
   assertion changes. Sampling may reduce redundant shapes/dtypes inside one mode;
   it must not remove a mode or invent a source-absent case. This sampling allowance
   applies only to correctness workloads; timing workloads always preserve the full
   core sequence from rule 2.
4. Definition.name is the real public callable, without a legacy flaggems_ prefix.
   Definition.parameters is the only ABI list. Preserve every parameter's name,
   order, kind, required status, and normalized default. reference.run() and an
   optional correctness_reference.run() must exactly mirror it. Workload.call must
   reproduce source positional arguments, explicit keywords, omitted defaults, and
   *args expansion. Do not use **kwargs to hide modeled public parameters.
5. Put stable mutation and return aliasing in Definition.effects. A validator must
   completely reproduce pytest's return-value rule for stochastic, NaN/Inf,
   partial-output, or special-tolerance cases. The v6.2 packager uses it instead of
   default return-value comparison; KGS separately enforces
   ABI/effects/shape/dtype/output-structure checks.
   correctness_reference.run() must preserve the candidate's complete structural
   contract: output count, shape, dtype, mutation, and aliasing. In particular, an
   upcast oracle for an in-place operator must copy the cast-back result into the
   original mutated parameter and return that same object; an out-of-place upcast
   oracle must cast its result back to the public output dtype. custom valid() runs
   after these structural gates and cannot repair a mismatch there.
6. Workloads contain JSON only. Use random for ordinary tensors, scalar/literal for
   directly representable values, and custom + generator_params for correlated
   tensors, layouts/views, Tensor containers, Generator, special scalar values, or
   framework objects. The optional fixed hook is gen_inputs(ctx, device); there is
   no hook-name field. Read recipes from ctx['inputs'][input_name] and the seed from
   ctx['seed']; gen_inputs returns a mapping keyed by public Definition parameter.
   Build custom
   random tensors from a seed-fixed local CPU Generator before moving to device.
   Preserve List versus Tuple identity and the exact forward-produced data flow.
7. Keep reference.run() as the raw original-dtype Torch/ATen benchmark operation.
   Put pytest-only CPU/upcast/formula work in correctness_reference.run(), casting
   results back exactly as the source assertion does. Convert reduce_dim=K to
   tolerance.atol_scale=K; emit rtol/atol only for explicit source overrides.
8. Keep correctness workloads representative and all workloads executable.
   Timing completeness is defined by the exact core sequence, not representativeness.
   API/capability failures are
   eligibility/unsupported results, not permission to change semantics. Before
   returning, audit every field against source and apply the reference-as-solution
   invariant. Reference sources run on a standalone evaluation Server and must not
   import flag_gems or rely on its runtime globals. Never weaken validation merely
   to obtain a pass.
""".strip()


OPERATOR_EXTRACTION_GUIDANCE = {
    "alpha_dropout": (
        "Match pytest literally. For train=True it checks only output shape and "
        "dtype, so valid() should return true after the Server's structural gate; "
        "do not invent distribution, mean, standard-deviation, or elementwise "
        "checks. For train=False pytest compares the candidate with a reference "
        "copy made from that same candidate, so it likewise adds no value check. "
        "Keep p and train in the named inputs mapping."
    ),
    "bucketize": (
        "x keeps the parametrized candidate dtype and boundaries is float32 because "
        "pytest omits dtype. Keep reference.run() as raw torch.bucketize and put the "
        "pytest x-to-float32 transform in correctness_reference.run(). Emit only the "
        "source-backed portable float32 timing subset."
    ),
    "max_pool3d_with_indices": (
        "Keep the benchmark's two-output (out, indices) interface and "
        "return_indices=True. Pytest compares only output[0]; a custom validator may "
        "ignore index values but must not bypass output structure/metadata gates."
    ),
    "gcd_": (
        "Correctness uses pytest randint(1,100); timing uses the benchmark's generic "
        "full-range integer inputs. effects must declare the first input mutated and "
        "the returned output aliased to that input."
    ),
    "igammac_": (
        "Correctness uses positive rand()+0.1; timing uses generic float32 inputs. "
        "Definition.name is igammac_. Keep timing tensors at most 16,777,216 elements. "
        "Put the pytest-only higher-precision computation in "
        "correctness_reference.run(), copy back into the original first input, and "
        "declare mutation/return alias in effects."
    ),
    "lcm": (
        "Correctness uses pytest randint(1,100); timing uses generic full-range integer "
        "inputs. Emit at most six workloads per phase and exclude billion-element "
        "stress shapes."
    ),
    "lcm_": (
        "Use the same source distributions as lcm. effects must declare the first "
        "input mutated and the returned output aliased to it."
    ),
    "linalg_svdvals": (
        "Correctness uses deterministic well-conditioned matrices and a higher-precision "
        "correctness_reference.run() that casts back to float32. Timing uses benchmark "
        "randn float32 matrices, not the correctness generator."
    ),
    "rnn_relu": (
        "input and hx use randn; w_ih, w_hh, b_ih, and b_hh use the source RNN "
        "uniform[-1/sqrt(hidden_size),1/sqrt(hidden_size)] initialization."
    ),
    "special_erfinv": (
        "Correctness uses uniform(-0.9,0.9) and its FP16/BF16 reference upcast in "
        "correctness_reference.run(). Timing keeps the UnaryPointwiseBenchmark input."
    ),
    "special_logsumexp": (
        "The implementation accepts dim as either an int or a list. Do not let a "
        "custom generator materialize List[int] for a parameter declared int: the "
        "Server validates materialized values against Definition.parameters. Emit "
        "two records with the same public Definition.name special_logsumexp and "
        "distinct record_id values: an int specialization containing the core timing "
        "sequence and scalar-dim pytest cases, and a correctness-only List[int] "
        "specialization containing the multi-dimension pytest mode. Keep both run() "
        "signatures named (inp, dim, keepdim=False); only the dim ABI type differs. "
        "Core timing must pass literal integer 1, never [1]."
    ),
    "addmm_": (
        "effects must declare mutation of self and the output alias to self. If the "
        "pytest oracle is upcast, correctness_reference.run() must compute in the "
        "oracle dtype, copy the cast-back result into the original self, and return "
        "self. Returning a newly cast tensor violates the public in-place ABI."
    ),
    "concatenate": (
        "Reproduce gen_cat_shapes_dim exactly. Every generated tensor shape must "
        "match on all axes except the normalized concatenation axis; pay particular "
        "attention to the source permutation for positive and negative dimensions."
    ),
    "log_normal_": (
        "The core benchmark contains large FP16 tensors where a mathematically valid "
        "log-normal tail can overflow to inf. Keep the pytest's positivity and mean "
        "assertion exactly for its accuracy workloads, using the source-visible "
        "explicit mean/std inputs to distinguish that mode from core timing, which "
        "omits the defaults. Make the separate timing-gate statistical "
        "check robust to representable FP16 tail overflow, for example by checking "
        "finite log-domain samples. Do not compare two independent random draws "
        "elementwise."
    ),
    "randint_like": (
        "A custom validator reads high from inputs['high']; workload recipes remain "
        "available in ctx['inputs']. Preserve the full public factory "
        "kwargs in the Definition even when source workloads omit their defaults."
    ),
    "softmax": (
        "Pytest upcasts only the oracle input. correctness_reference.run() must cast "
        "the oracle output back to the public output dtype (float32 only when "
        "half_to_float requests it), because the Server checks dtype before custom "
        "numeric validation. Preserve negative-infinity generation and equal_nan."
    ),
    "special_modified_bessel_k0": (
        "Do not turn to_reference(upcast=True) into unconditional float64. Use the "
        "source-backed portable float32 subset and positive rand()+0.1 correctness "
        "inputs; reference.run() is also the correctness path for that subset."
    ),
    "special_chebyshev_polynomial_u": (
        "The benchmark covers scalar n=3. Keep the benchmark-backed callable name "
        "special_chebyshev_polynomial_u and make any source pytest tensor-n overload "
        "correctness-only. Use outer record_id values to distinguish specializations; "
        "do not invent an overload suffix as a public symbol."
    ),
    "special_shifted_chebyshev_polynomial_w": (
        "Correctness uses random float32 x and custom int32 randint(0,11) tensor n; "
        "the higher-precision oracle belongs in correctness_reference.run(). Timing "
        "uses two random float32 tensors. Preserve the source NVIDIA-only eligibility."
    ),
    "unbind_copy": (
        "Use Definition.name unbind_copy and preserve its actual public signature. "
        "The single logical output is the variable-length List[Tensor] returned by "
        "torch.unbind_copy; do not split by arity, stack, or wrap it."
    ),
    "einsum": (
        "Use Definition.name einsum and preserve the actual variadic public signature; "
        "do not reuse the old synthetic (equation, operands) ABI. The Workload call "
        "must reproduce the source expansion. Operand metadata is a custom JSON recipe "
        "with independent item shapes/dtypes, materialized by gen_inputs(ctx, device). "
        "Put the pytest FP32-copy oracle in correctness_reference.run()."
    ),
    "unsqueeze_": (
        "Read only test_unsqueeze_ cases. Declare the in-place mutation and returned "
        "alias in effects. Timing repeatedly receives the same input, with no reset."
    ),
    "thnn_fused_lstm_cell_backward_impl": (
        "Construct cx, cy, and workspace from the same aten forward invocation in "
        "gen_inputs; create gradients afterward. Keep has_bias=True only."
    ),
    "fused_adam_": (
        "Reproduce source single-element TensorLists and exact zero/state-step "
        "initialization. Keep amsgrad=False and maximize=False. Declare all stable "
        "mutations/aliases in effects, while preserving the pytest's compared outputs."
    ),
    "thnn_fused_lstm_cell": (
        "Use only LSTM_SHAPES, preserve no-bias and two-random-bias correctness modes, "
        "and the no-bias timing mode. Pytest compares hy/cy but not workspace values."
    ),
    "scaled_dot_product_attention_backward": (
        "Use the legacy backward pytest, source LEGACY_SHAPES, enable_gqa, source "
        "uniform(-0.05,0.05), and forward/autograd data flow. Preserve equal_nan and "
        "dV tolerance. No benchmark exists, so timing is empty."
    ),
}


# V6 deliberately has no ABI-bearing few-shot shells. Hard-coded signatures can
# fossilize the exact mismatch this extractor is required to reject.
OPERATOR_DEFINITION_SHELLS: dict[str, dict] = {}
