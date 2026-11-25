# Interfaces

Install from `hgkda/` with `python -m pip install .`. The skill script and
`python -m hgkda` use the same entry point. Run examples from `hgkda/`:

```sh
python -m hgkda doctor
python -m hgkda compare --directory examples/paired-expm1/paired --output /tmp/comparison.json
python -m hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison /tmp/comparison.json --output /tmp/context.json
python -m hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py
python -m hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json
python -m hgkda timeline --listing <listing.json> --trace <raw-trace.json> --binding <binding.json> --output <relative-timeline.json>
python -m hgkda knowledge --operator expm1_
```

On Windows replace `/tmp/` with a new external output directory. Outputs never overwrite
existing evidence. The `hgkda.result/v1` envelope includes `operation`, `status`, `inputs`,
`result`, and SHA-256 for every input. See the [schema](../../../schemas/result.schema.json).

`compare` reads both sides' `collection.json` and CSV under the same `spec.json`.
`hotspots` requires `source_sha256` in the listing. Import historical listings explicitly
after verifying capture requests/source, never by filename alone. `context` separates
`facts` (comparison, source/IR, Roofline, hotspots, regions), `hypotheses` (heuristics),
and versioned `knowledge`. It neither invokes the model nor asserts optimization occurred.

## Timelines and regions

`regions --listing <listing.json> --trace <bound-trace.json> --kernel <symbol>
--core <core> --trace-unit us` accepts an instruction timeline for one kernel and core.
`hgkda_binding` declares matching `source_sha256`, `case`, `kernel`, `core`, `time_unit`.
The collection adapter verifies raw identity/scope. Confirm units from collection
format, not `displayTimeUnit`. For each pipeline, interval unions partition the same
window into `productive_union`, `wait_only_union`, and `idle_union`. Different pipelines
can overlap; do not add them into total runtime.

Send the result through `context --listing <listing.json> --regions <regions.json>`
with same-source/case profiling and IR. Idle means no observed execution in the window;
it does not alone establish stall causality or a critical path.

`timeline` supports a common-core clock whose absolute unit still needs verification.
The binding declares `source_sha256`, `case`, `kernel`, `core`, `raw_trace_sha256`,
verified against the raw trace and listing. Output uses `time_unit=fraction`,
`clock_qualification=relative_only`, `absolute_time_qualified=false`, and window [0,1].
Fractions are invariant to positive clock scaling. Execution, explicit WAIT/BAR waits,
and observed idle partition each reported lane. Metadata-only lanes retain idle;
unreported lanes remain unknown. Preserve partial captures and clock qualifications.
Use `context --regions` to pass normalized results; verify absolute durations separately.

Synchronization CSV cumulative time and trace intervals have different scopes. Use
non-synchronization instructions for clock checks; keep waits in region/timeline analysis.

## Numerical reference

`roofline --input <json>` accepts verified GEMM work and matching collection/reference:
`facts`, `shape` (M,N,K), `dtype`, `empirical`, with `source_sha256` and `case`.
Without a workload contract, use paired profiling rather than estimated substitutes.

## Knowledge and screening

`curate --records <records.json> --root <evidence-root>` deduplicates by operator type,
action, and scope. `validated` requires matching source, full request/result, backend
review, and required supplementary identities. Missing evidence downgrades to `draft`;
replaced records remain `superseded`.

`observed_failure` includes supplementary failures or correctness-pass/performance-fail
counterexamples. The latter binds full same-source results, feedback, and
`scope.performance_threshold`. Prose alone is insufficient.

`retrieve --index <index.json> --operator-type <type> --scope <scope.json> --max-chars 6000`
revalidates evidence and retrieves scoped whole records without truncating individual
entries. These are deterministic rules, not automatically learned successful methods.

`gate --source <candidate.py>` screens known paths; `--review` binds an existing review.
KGS evaluation also needs a frozen request/review and fresh output directory. See
[backends](../../../docs/backends.md).

## Feedback and runtime

`feedback --operator <name> --source <candidate.py> --result <result.json> --audit
<audit.json> --timing-cases <timing.jsonl> --output <new.json>` verifies source, audit,
and case identity, reconstructs geometric mean, selects loss cases, and covers the
largest input per dtype. `continue_optimization` returns case feedback; `target_met`
requires exceeding the threshold (default 0.8x) and required checks. Offset/stream
checks remain separate. Feedback does not invoke models/devices or replace host review.

`python -m hgkda.runtime --help` exposes configured prepare/context/preflight/generate/
pair/simulate/evaluate actions. The host owns orchestration and stop conditions.
See [public runtime](../../../docs/runtime.md).
