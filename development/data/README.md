# Hardware metrics and statistics

| File | Purpose |
| --- | --- |
| [hardware-metrics-comparison.xlsx](hardware-metrics-comparison.xlsx) | Final cross-vendor metric comparison |
| [hardware-statistics.xlsx](hardware-statistics.xlsx) | Ascend specifications/evidence with other vendor data retained |
| `archive/` | Original team spreadsheets from November 21/22 |

Workbook contents and bytes are unchanged. The [November 23 map](relocation-20261123.json)
preserves the move from repository root; the [November 24 map](../manifests/engineering-relocations-20261124.json)
records English names for the two current workbooks. Historical spreadsheet identifiers
remain exact in these maps. Presentation workbooks are display copies; this directory
is the maintenance location.

Evidence: [metric comparison](../reports/indicator-comparison-20261121/README.md) and
[Ascend statistics](../reports/ascend-statistics-20261120/README.md). Frozen reports/logs
retain historical paths. Resolve them with `python -m development.tools.maintenance.resolve_path
--path <historical-repository-path>`.
