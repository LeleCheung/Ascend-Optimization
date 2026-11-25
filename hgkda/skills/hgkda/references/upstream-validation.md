# Native FlagGems acceptance

For upstream delivery, run native tests/benchmarks in addition to full KGS and HG-KDA
supplementary checks. Preserve each layer separately.

## Workspace and responsibility

Use an explicit external Git workspace pinned by `upstream_commit`. Never modify the
frozen internal snapshot or require `development/` at runtime. CC/DeepSeek owns candidate
integration, dispatch, launch wrappers, and implementation changes; Codex maintains
adapters, analyzes failures, and updates methods.

Keep original delivery SHA, model I/O, integration patch, actual imports, commands,
and raw results. Integrated source gets a new identity and cannot inherit earlier
KGS acceptance. If a module must support both KGS and FlagGems, validate exported
symbols: KGS requires `run` and operator exports to be the same function object.
Let the generating model adapt without weakening loading checks or creating alias recursion.

Run exact-operator correctness/benchmark first, then related regression and target CI
as required by scope. Selected tests are separate from full repository CI or PR review.

## Configuration and entry

```sh
python -m hgkda.runtime upstream-test --config /path/upstream-test.json
```

Configuration includes `repository`, `upstream_commit`, `operator`, `integration_files`
(relative paths mapped to SHA256), `expected_function_source`, source-bound strict
`source_review`, `delivery_source`, `delivery_sha256`, `python`, `sdk`, `device`, `lock`,
`output`, and optional `container`, `environment`, `timeout_seconds`. Use a fresh output
directory outside the workspace and exclude credentials. Only declared `src/flag_gems/`
files may change; tests, benchmarks, accuracy requirements, and config stay pinned.
Honor exclusive device locks.

Set `require_kgs_entrypoint=true` when the same file must support KGS. Import preflight
checks function identity before expensive tests. The adapter verifies actual imports,
then invokes native `tools/run_tests.py --ops <operator>` without quick mode. Upstream
`--stages` filters operator maturity, not accuracy/performance phases. Preserve core
benchmark results, skips, and collection errors.

Default `mode=scheduler` uses the native scheduler. If unrelated collection errors
block it, preserve failure and explicitly select `mode=operator-files` with complete
`accuracy_files`, `benchmark_files`, optional `marker`, and `accuracy_reference`.
Run unchanged native files and parsing. `expected_accuracy_cases_from` checks the
original selected case set. Keep modes distinct; exact-file success is separate from
whole-scheduler success.

## Results

`result.json` retains passed counts, skips, dtype/shape benchmark rows, and geometric
mean. Require actual passed cases and valid timing; zero exit, all skipped, missing
results, or collection errors cannot pass. Native reference/timing follows upstream
and is separate from fixed KGS/CANN scores. Diagnose candidate, integration, and
environment/infrastructure failures without deleting cases or relaxing tolerances.

## Launch/profiler compatibility

Ordinary timing success does not establish profiler compatibility. Raw integer pointer
arguments may fail if the active vendor launcher queries `tensor.size()`: historical
log10_ passed 18 accuracy cases but failed profiler launch this way. Compare parameter
contracts and separate host/collection interfaces from device math.

The generating model should preserve required Tensor metadata or use supported Triton
launching. Keep profiling/timers enabled and computation unchanged; never replace
candidate math with vendor compute. Revalidate native benchmarks and new-source KGS/
supplementary checks after fixes.

Always passing Tensor arguments may increase small-input host cost. Inspect per-case
latency and actual profiler-state interfaces. Different representations may satisfy
ordinary/profiling contracts while executing identical computation. Test calls after
profiling ends; vendor flags may remain active. Never retry an in-place launch after
an exception unless non-execution is established. Dispatch cannot depend on test names,
answers, or result files.

Check ordinary calls, after unrelated reference-only profiling, during candidate
profiling, and after collection, across dtypes with in-place identity/numerical checks.
Profiler-state queries are host metadata only. Pin verified internal interface versions
and retain a metadata-safe path when state is unknown.
