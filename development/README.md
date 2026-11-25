# Development and research

The November 2026 presentation phase is complete. Current engineering work covers
repository organization, English source/comments/runtime text, and maintainable
documentation. Documentation can be English or bilingual English/Chinese.

## Directory map

| Directory | Contents |
| --- | --- |
| `tests/hgkda/` | Offline package, runtime, identity, and distribution tests |
| `tests/` | Research-tool and presentation-data regression tests |
| `tools/maintenance/` | Repository inventory, integrity, and maintenance checks |
| `tools/skill_validation/` | Method freezing, generation audits, and evidence export |
| `tools/regression/` | Batch acceptance and result audits |
| `tools/backends/`, `tools/operators/` | Internal reference-building and operator support |
| `tools/traces/`, `tools/lineage/` | Trace ingestion, provenance, and views |
| `tools/research/` | Research orchestration and compiler experiments |
| `tools/archive/` | Historical tools retained for reproducibility |
| `presentation_archive/` | Byte-exact completed presentation generators and templates |
| `data/` | Current and archived specification/metric spreadsheets |
| `runbooks/` | Host, container, device, and evaluation procedures |
| `manifests/` | Source identities, relocation maps, and maintenance inventories |
| `reports/` | Research results, presentation artifacts, and engineering reports |
| `experiments/` | Frozen original experiments; preserve exact bytes |
| `dependencies/` | Pinned third-party snapshots; preserve upstream contents |

See the [tool index](tools/README.md) for entry points.

## Software workflow

Run from the repository root:

```sh
python -m pip install ./hgkda
python -m unittest discover -s development/tests/hgkda -v
python -m unittest discover -s development/tests -p 'test_*.py' -v
python -m development.tools.maintenance.audit_language
python -m development.tools.maintenance.verify_layout --repository . --output /path/new-layout-audit.json
python -m hgkda.runtime --help
python -m development.tools.skill_validation --help
```

Software tests are independent of device acceptance and normal skill calls.
NPU experiments require the configured Linux environment; see the
[910B runbook](runbooks/910b-testing.md). The runtime must not import or require this
directory. `tests/hgkda/test_public_distribution.py` checks isolated distribution.

## Results and entry points

- [Operator index](../operators/index.json): 26/30 main-line deliveries and nine
  earlier records, with separate noncompliant, strict historical, and independent versions.
- [GQA goal records](reports/gqa-goal-20261121/README.md): retained source/IR,
  negative results, and evaluation lineage.
- [GQA presentation evidence](reports/gqa-meeting-traces-20261122/four-case-results.md):
  case 4 and four-case geometric-mean results; the full 20-case set remains future work.
- [Three-part migration](reports/three-part-migration-20261111.md).
- [Spreadsheet index](data/README.md).
- [Architecture](docs/architecture.md) and [language policy](docs/language-policy.md).
- [Engineering cleanup](reports/engineering-cleanup-20261124/README.md).

Resolve the older three-root layout using `manifests/three-part-migration-20261111.json`.
Current moves are recorded in `manifests/engineering-relocations-20261124.json`;
`python -m development.tools.maintenance.resolve_path --path <old-path>` verifies and
resolves presentation/spreadsheet relocations, including the prior spreadsheet move.
Do not rewrite raw requests, snapshots, or measured results to look current.
The previous chronological landing page remains in
[originals](reports/engineering-cleanup-20261124/originals/development/README.md).
