# Internal method validation

`prepare-history`, `audit`, `finalize`, `export`, and `verify` manage upstream starts,
method snapshots, model-generation provenance, and archive integrity. These tools
are not public skill runtime dependencies.

Device capture, context, model calls, preflight, and evaluation use the installed
`python -m hgkda.runtime --help` interface. Internal actions are listed by
`python -m development.tools.skill_validation --help`.

Historical snapshots keep original paths/SHA. Resolve them through
`development/manifests/three-part-migration-20261111.json` without rewriting originals.
