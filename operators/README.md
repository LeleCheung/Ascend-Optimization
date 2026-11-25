# Operator deliveries and versions

[index.json](index.json) records 30 main-line operators and nine earlier records.
**26 main-line operators are accepted.** Follow each manifest's `selected` field to
the exact retained source; incomplete candidates are not accepted deliveries.

Version classes are `historical_noncompliant`, `historical_strict`, and
`independent_strict`. The last distinguishes retained delivery regression from
model-generated candidates. Missing originals remain explicitly unavailable.

| Directory | Purpose |
| --- | --- |
| `versions/` | Original candidate bytes and bound reviews |
| `contracts/` | Formal cases, references, precision and timing requirements |
| `references/` | Supplementary PyTorch/CANN references and native resources |
| `checks/` | Delivered supplementary and operator-specific validation |
| `evidence/` | Full results and provenance for selected versions |

`relocation.json` explains contract/reference moves. Historical absolute request
paths are provenance; runtime preparation creates new current configuration.
Keep hash-bound resources unchanged during engineering/language cleanup.

Native CANN libraries depend on [environment.lock.yaml](environment.lock.yaml) and
are not assumed portable across Python ABIs or CANN versions. Repacked references
receive new bundle hashes; historical scores stay bound to original contracts.
Use a new output directory for every run.

See the [public runtime guide](../hgkda/docs/runtime.md). Public execution requires
the package, selected delivery, and configured external services. Git LFS supplies
native assets. Earlier operators remain historical records, not new experiment targets.
