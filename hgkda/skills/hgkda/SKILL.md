---
name: hgkda
description: Diagnose NPU operator performance with paired profiling, bound source/IR, numerical references, and simulator hotspot regions; guide testable strict Triton optimization and full optional KGS evaluation.
---

# HG-KDA operator diagnosis and optimization

Use the standalone Python core to compute evidence. The calling host supplies the
model that explains mechanisms and generates candidates. KG is optional; KGS provides
full evaluation; msprof provides device collection.

Codex maintains diagnosis, workflow, skills, knowledge, analysis, and scheduling.
CC/DeepSeek writes all target candidate code, including launch wrappers and fallbacks.
Do not implement candidate patches indirectly in control scripts. Preserve model
inputs/outputs, method versions, source SHA, and formal evidence.

## Start

Install with `python -m pip install .` from `hgkda/`, then run
`python skills/hgkda/scripts/run_hgkda.py doctor`. Outputs use stable JSON envelopes;
`--output` requires a new file. See [interfaces](references/interface.md).

Fix operator, input, source SHA, contract, reference provenance, and device. Historical
captures are not current-source evidence. If the starting candidate contains non-Triton
device code, ask CC/DeepSeek for a minimal compliant replacement before collection.
Run path/address preflight first, then collect actual IR and paired profiling for that
candidate. Preflight covers the submitted implementation, not every device action of
exploratory test scripts. Missing IR/simulation remains explicitly unknown.

## Staged workflow

1. **Establish facts.** `compare --directory <paired>` preserves distributions, valid
   counter totals, ratios, and average transfer granularity. Do not fill missing data.
2. **Build context.** `context --operator <name> --source <candidate.py> --comparison
   <comparison.json>` binds identity and separates evidence, hypotheses, and knowledge.
   Include actual IR, Roofline, and instruction listings only when bound and available.
3. **Resolve ambiguity.** Rank instructions with `hotspots`, then inspect source regions
   with `regions`. Bound instruction timelines identify execution, explicit waits, and
   observed idle on one source/case/kernel/core. Provide these via `context --regions`.
   Examine data movement, compute, writeback, and IR; a high-cost instruction alone
   does not establish cause. After simulation is introduced, inspect both hotspot and
   pipeline evidence after each complete candidate, recording when either has no new
   diagnostic role. Keep immutable captures and raw formats. If common-core timing
   exists but absolute units remain unresolved, use `timeline` for normalized fractions,
   preserving `relative_only` and partial-window scope; never invent missing timing.
4. **Design a discriminating experiment.** Record evidence, hypothesis, alternatives,
   change, expected observations, validation, and unresolved questions. Prefer one
   mechanism per experiment and verify generated-code behavior.
5. **Validate and continue.** Review every candidate path, run the full KGS contract and
   supplementary checks, then return full results to the model. Default to performance
   first without invented call/time/token/cost budgets. Meeting the threshold does not
   stop optimization unless requested. Choose representative loss cases and continue
   after failures; one model return or transport error is not a stopping rule. Preserve
   negative results and deduplicate validated knowledge by scope.

Before a concrete edit, submit a `runtime decision` binding current source, evidence,
mechanism, and expectation. Tool reads are not automatic proof of decisive guidance.
See [continuous trace](references/trace.md). Keep failed and unchanged rounds as well
as successful ones; do not reconstruct missing historical pre-change decisions.

For similar fixed latency across small pointwise inputs, inspect the actual launch
path and feedback before changing device tiling. See [launch and feedback](references/launch-and-feedback.md).
When `knowledge` reports `retrieval.empty`, state the method source and distinguish
general constraints from retrieved operator-specific evidence.

## Contracts

Strict candidates use Triton for all device computation, copies, and fallbacks.
CANN/Ascend C is reference-only. Preserve strides, offsets, dtypes, in-place/out identity,
special values, current device, and streams. Never cache tensor data/results or old streams.

Use same-environment microbenchmarks as empirical envelopes, separate from physical
peaks. Overlapping pipeline durations cannot be added into total latency. Simulator
time is not formal hardware timing. Default to one full evaluation per source; explain
additional repeats rather than replacing full acceptance with local benchmarks.

For upstream delivery, also run native tests/benchmarks in an explicit external pinned
FlagGems workspace using `runtime upstream-test`; report all three layers separately.
See [upstream validation](references/upstream-validation.md).

Read versioned knowledge with `knowledge --operator <name>`. Keep facts, hypotheses,
and validated findings distinct. All device/model entry points are packaged in
`python -m hgkda.runtime`; delivery contracts/checks come from `operators/`. Installed
operation must not require internal `development/` tools.
