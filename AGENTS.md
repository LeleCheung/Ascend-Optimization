# HG-KDA maintenance rules

## Current phase and language

The presentation phase is complete (2026-11-24). Current work is engineering
organization and English source, comments, prompts, configuration descriptions,
and runtime text. Documentation may be English or bilingual English/Chinese.
Use UTF-8 and stable English identifiers. This supersedes the former Chinese-only
writing rule. See [language policy](development/docs/language-policy.md).

The exact previous instructions and chronological research log are preserved in
[the historical snapshot](development/reports/engineering-cleanup-20261124/originals/AGENTS.md).
Past optimization goals are completed historical records, not instructions to
restart device jobs during cleanup.

## Architecture

- `hgkda/`: installable standard-library diagnostics, public runtime, agent skill,
  knowledge, curated examples, and user documentation.
- `operators/`: delivered candidates, selected versions, contracts, references,
  supplementary checks, and bound acceptance evidence.
- `development/`: software tests, reusable engineering tools, research records,
  frozen experiments, reports, and pinned upstream snapshots.

The public package and deliveries must work in an isolated distribution without
`development/`. Historical source paths are provenance, never runtime dependencies.
Do not import KG/KGS internals or hard-code machine paths into the public package.
Keep environments, caches, credentials, logs, and active runs outside the repository.
Do not change or stage `.vscode/`.

## Identity and research integrity

Preserve frozen source bytes, hashes, formal results, contracts, tolerances, source
maps, failure records, and attribution. Never invent missing versions or overwrite
old evidence. Use a relocation map for historical paths when moving files.
Directory cleanup does not increase the accepted-operator count or authorize new
operator experiments. Treat third-party and frozen experimental source as records;
provide English navigation without changing their identity.

`operators/index.json` is the delivery authority (26 accepted of 30 main operators,
plus nine historical operators). Do not resume the earlier nine or extend the 26
operators merely to prepare presentations. GQA completed on 2026-11-22: full-call
120.9111 us versus CANN 127.9771 us, 1.05843963x; four-case geometric mean
1.035690x CANN. Case 4 drove optimization with continuous four-case validation;
full 20-case evaluation is separate future work. The retained best uses IR injection.
Preserve the distinction between full-call performance and independent kernel timing.

## Candidate authorship and execution boundaries

Codex maintains diagnosis, workflow, skills, knowledge, analysis, scheduling, and
evaluation infrastructure. CC/DeepSeek generates and modifies target candidates,
including launch wrappers, caches, indexing, layouts, fixes, fallback paths, and
candidate-specific IR/pass patches. Do not generate candidate patches indirectly
through control scripts or replace model output with a complete implementation
packaged as knowledge. Record evidence, applicability, hypotheses, proposed actions,
and verification before handing a method to the generating model.

Strict Triton deliveries use Triton for all device computation, copies, reductions,
scans, layout conversions, and fallback paths. Python/C++ handles allocation,
metadata, parameters, compilation caching, state, and launching. CANN/Ascend C is a
reference; native DMA, PyTorch target computation, or CPU computation cannot replace
a strict candidate. Capability-extension/custom-op research is separately identified
and is not counted as strict Triton delivery.

Fix reference, contract, accuracy, and timing scope. Record static screening,
source-bound path review, full KGS evaluation, and supplementary checks separately.
Default to one formal evaluation per source; explain additional repeats. Upstream
FlagGems acceptance uses an explicit external pinned workspace, unmodified tests,
and native benchmarks, separately from KGS and HG-KDA checks. Do not relax tolerances,
confuse selected-operator tests with full CI, or reuse old scores after source changes.

Windows is the control host; configured remote containers provide NPU environments.
Check SSH aliases and remote identity before declaring remote access unavailable.
Operate only on project-owned paths/processes; do not restart shared services or
containers. Engineering cleanup does not require remote device experiments.

## Evidence and figures

Separate facts, rule hypotheses, and validated conclusions. Record missing evidence;
thresholds prioritize investigation rather than prove causation. Do not sum overlapping
pipeline times or present simulator instruction time as a real-device critical path.
Preserve pre-change decisions, model/tool logs, actual source/IR, result bindings,
and negative results. Historical missing decisions remain missing.

Reports center on HG-KDA overview and GQA. Keep figures simple: necessary title,
axes, result, legend, and short method labels; no dense footnotes or defensive
statements. Inspect rendered figures. Complete traces omit method annotations;
compact traces retain original indices and use short arrows. Color denotes guidance
(blue paired profiling, orange simulator, purple knowledge/code reasoning); shape
denotes changed object (square Host, circle Triton, triangle IR, diamond Mixed).
Keep original attribution and joint contributions in data, never rewrite evidence
to fit the narrative. Historical presentation layout remains frozen unless requested.

## Maintenance and validation

Root README navigates the three roots. Package README documents installation and
stable interfaces. Developer README documents engineering entry points and status;
put chronological logs in reports. Keep maintained references current after moves.
Validate package tests, isolated distribution, source identity, layout, and language.
Do not introduce tests that only mirror trivial wording; retain substantive checks.
