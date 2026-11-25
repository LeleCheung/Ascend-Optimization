# Completed presentation generators

The November 2026 presentation phase is complete. This directory retains 33 historical
presentation/fixed-report generators and their HTML template with their exact original bytes and languages.
They are reproduction records rather than maintained English engineering interfaces.
Existing PPT/PDF/HTML results and research evidence remain in their original locations.

- `research/`: dated slides, animations, fixed experiment reports, and workbook generators.
- `traces/`: GQA/meeting slide and presentation-view generators.

The [relocation manifest](../manifests/engineering-relocations-20261124.json) binds every
old/new path and SHA-256. Directory depth is preserved for original root calculations.
Small English bridge modules retain relative imports of maintained trace helpers;
the archived implementations themselves are unchanged.

Use `python -m development.tools.maintenance.resolve_path --path <old-path>` to locate
an artifact and verify its identity. Old report links/commands remain historical;
resolve their paths rather than editing original reports. Exact reproduction should
use the recorded source revision and method/environment snapshots, since current
analysis helpers and English labels have evolved. These generators may write report
outputs when executed; reviewing the archive does not require running them.

Maintained tools and interfaces are indexed in [tools/README.md](../tools/README.md).
