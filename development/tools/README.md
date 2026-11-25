# Engineering tools

| Directory | Purpose | Entry points |
| --- | --- | --- |
| `maintenance/` | Layout, identity, language, and repository checks | `inventory.py`, `verify_layout.py`, `audit_language.py`, `resolve_path.py` |
| `regression/` | Fixed-candidate preparation, full evaluation, offline audits | [Regression guide](regression/README.md) |
| `skill_validation/` | Method identity, generation provenance, transfer/archive checks | [Method validation](skill_validation/README.md) |
| `backends/`, `operators/` | Optional native references and operator-specific checks | [Backend guide](backends/README.md) |
| `lineage/`, `traces/` | Read-only evidence ingestion, source ancestry, and derived views | Original evidence stays frozen |
| `research/` | Explicit research orchestration, compiler capture, and IR review | [Research guide](research/README.md) |
| `archive/` | Earlier retired tools | Historical source; do not use as default controllers |

Completed presentation generators live in [presentation_archive](../presentation_archive/README.md),
separate from reusable engineering tools. Their original bytes and previous paths
are recorded in the [relocation manifest](../manifests/engineering-relocations-20261124.json).

Run module entry points from the repository root. Public device/model actions use
`python -m hgkda.runtime`; internal tools are not dependencies of installed users.
Offline tests use the standard library. Optional rendering/workbook dependencies
include Playwright/Edge, python-pptx, openpyxl, xlsxwriter, Pillow, and lxml;
install them only for the relevant internal workflow.
