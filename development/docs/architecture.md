# Repository architecture

## Public package

`hgkda/src/hgkda/` provides a standard-library-only offline core. `cli.py` dispatches
diagnostics. `compare`, `hotspots`, `regions`, `timeline`, and `roofline` turn bound
inputs into structured evidence. `context`, `curation`, and packaged knowledge prepare
model inputs. Facts, hypotheses, and validation remain separate. The model instruction
template is packaged as `data/prompts/agent.md`, independently of runtime Python code.

`adapters/` connects explicit external services; `runtime/` provides individual
host/device actions. Importing the package does not connect to services/devices.
The caller controls model invocation, review, and iteration. Runtime configuration
supplies environment identities rather than importing internal research tools.

`skills/hgkda/` is the agent workflow/reference. `docs/` explains integration;
`schemas/` defines result envelopes. `examples/` holds selected immutable captures
and sources with original provenance.

## Operator deliveries

`operators/<name>/manifest.json` selects exact source and records versions,
contracts, checks, references, and evidence. Native reference libraries are
environment-specific Git LFS assets. Repackaging creates a new bundle identity;
it never rebinds an old score to new bytes.

## Development layer

Reusable maintenance, research, trace, and audit tools live in `development/tools/`.
Tests stay in `development/tests/`. Generated reports, frozen experiments, and
pinned upstream trees have separate homes and are not runtime dependencies.

The root README is a navigator; package docs serve users; the developer guide serves
maintainers. Detailed histories belong in reports, not current landing pages/rules.
Completed generators are preserved in `development/presentation_archive/` with
exact hashes and a relocation map. Final presentation documents stay in their report
workspace. Current hardware workbooks live in `development/data/` under English names.
