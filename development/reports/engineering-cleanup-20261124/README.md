# Engineering organization and English-language maintenance

Completed 2026-11-24, after the presentation phase.

## Result

The project retains three engineering roots with explicit responsibilities:

| Root | Responsibility |
| --- | --- |
| `hgkda/` | Installable diagnostics, public runtime, agent skill, knowledge, and user guides |
| `operators/` | Exact delivered versions, contracts, references, checks, and acceptance evidence |
| `development/` | Maintained tools/tests/runbooks, historical research, and pinned dependencies |

Start at [root README](../../../README.md), [developer guide](../../README.md),
[tool index](../../tools/README.md), and [architecture](../../docs/architecture.md).

## Changes

- Converted maintained Python/C++ comments, docstrings, CLI/errors, diagnostic prose,
  model prompts, configuration descriptions, skill references, and current guides to English.
- Moved the public agent prompt into packaged `hgkda/data/prompts/agent.md`; conditional
  workflow behavior and installed access are verified. Removed unused nonexistent
  diagnostic-reference names while preserving knowledge selection.
- Replaced accumulated historical instructions on landing pages with concise English
  guidance. [Five original documents](originals-manifest.json), including full AGENTS.md,
  remain byte-preserved in `originals/`.
- Separated 32 completed presentation/fixed-report scripts and one HTML template into
  [presentation_archive](../../presentation_archive/README.md). Small bridge modules
  preserve relative imports of maintained trace analysis. Original archived bytes remain exact.
- Renamed the two current spreadsheets to `hardware-metrics-comparison.xlsx` and
  `hardware-statistics.xlsx` under [data](../../data/README.md), preserving workbook bytes.
- Added a [relocation manifest](../../manifests/engineering-relocations-20261124.json)
  for all 35 moved artifacts. The read-only resolver handles their old paths and the
  earlier spreadsheet relocation without modifying frozen references.
- Added a declared [language scope](../../manifests/english-language-scope.json),
  an English-language audit, relocation checks, and regression coverage. Current code
  still accepts original Chinese table headers through explicit legacy aliases.
- Fixed existing offline research-test fixtures missing compiler verification/context
  and deferred Linux-only device locking imports so offline helpers load on Windows.
  Full identity/evidence assertions remain enforced.

[Tracked changes and new engineering files](tracked-changes.json) provide an inventory.
Pre-existing untracked presentation/research work stays outside this cleanup inventory.

## Preservation and language boundary

Maintained engineering code, comments, prompts, and runtime text are English. Documents
may also be bilingual. Frozen candidates, raw captures, contracts, model logs, numerical
results, upstream snapshots, and archived presentation sources retain original bytes
and language. Their English navigation is separate. Six historical knowledge source
path/hash identifiers remain exact; they are provenance, not runtime file dependencies.
See [language policy](../../docs/language-policy.md).

The original example README is itself bound by the previous migration manifest;
its English working guide is [examples](../../../hgkda/docs/examples.md).
Current accepted delivery status remains 26/30 plus nine historical records. No new
NPU experiments or changes to research scores were part of this engineering work.

Exact historical presentation regeneration uses its recorded revision and method/
environment snapshot. Current analysis labels and prompts have changed language;
the archived generators are reproduction records, not current default entry points.

## Validation

| Check | Result |
| --- | --- |
| Core/runtime/public-distribution tests | 101 passed |
| Research/maintenance tests | 62 passed |
| Maintained language scope | 320 files passed; six exact provenance identifiers retained |
| Original migration integrity | 33,503 immutable files verified |
| Delivered identities | 527 artifacts across 39 operators verified |
| Relocations | 35 current moves plus four earlier spreadsheet mappings verified |
| Python source parsing | 280 maintained files parsed |
| C++ reference bridge | Changes confined to comments |
| Installed wheel | Built and installed outside checkout; doctor, runtime help, offline comparison, packaged prompt passed |
| Maintained Markdown links / diff whitespace | Passed |

Evidence: [summary](validation-summary.json), [language](language-validation.json),
[layout](layout-validation.json), [relocations](relocation-validation.json),
[distribution](distribution-validation.json), [source](source-validation.json),
[links](link-validation.json).

Reproduce offline checks from repository root after installing `./hgkda`:

```sh
python -m unittest discover -s development/tests/hgkda -v
python -m unittest discover -s development/tests -p 'test_*.py' -v
python -m development.tools.maintenance.audit_language
python -m development.tools.maintenance.verify_layout --repository . --output /path/new-layout-audit.json
python -m development.tools.maintenance.resolve_path --path development/tools/traces/gqa_intro_slide.py
```

Use a new output path for every audit. Public installation remains independent of
`development/`; device acceptance requires its configured Linux/NPU environment.
