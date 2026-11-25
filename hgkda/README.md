# HG-KDA

Hardware-Grounded Kernel Design Agent diagnoses Ascend operators using paired
hardware profiling, source and compiled IR, instruction hotspot regions, pipeline
timelines, and versioned knowledge. An agent skill guides the model; a standalone
Python core computes evidence; optional adapters connect to collection and evaluation.
Offline analysis uses the Python standard library. The host supplies the model.

## Install

Requires Python 3.10 or newer. From this directory, preferably in a virtual environment:

```sh
git lfs pull
python -m pip install .
python -m hgkda doctor
```

LFS supplies original example archives. Installation does not install KG/KGS,
FlagGems, a model client, or the NPU stack. See [optional backends](docs/backends.md).

## Offline examples

Run from this directory and choose new output paths outside the repository:

```sh
hgkda compare --directory examples/paired-expm1/paired --output ../hgkda-output/compare.json
hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison ../hgkda-output/compare.json --output ../hgkda-output/context.json
hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py --output ../hgkda-output/hotspots.json
hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json --output ../hgkda-output/regions.json
```

These produce source-bound comparisons, model context, hotspot rankings, and source
regions. Existing outputs are not overwritten. See [examples](docs/examples.md),
[JSON envelope](schemas/result.schema.json), and [CLI reference](skills/hgkda/references/interface.md).

## Capabilities and workflow

| Capability | Output |
| --- | --- |
| Paired profiling | Per-kernel distributions, counter totals, candidate/reference ratios, movement granularity |
| Staged diagnosis | Mechanism hypotheses, missing evidence, and next investigations |
| Instruction and region analysis | Hotspot rankings, source regions, execution/wait/idle intervals |
| Roofline reference | Contract-matched work, traffic, empirical envelopes, and physical peaks |
| Context and knowledge | Source-bound evidence, scoped retrieval, and validated deduplication |
| Evaluation | Strict Triton screening, reviewed KGS requests, full results, and performance feedback |
| Upstream acceptance | Explicit external FlagGems import checks, native correctness tests, and benchmarks |

Fix source and contract, collect paired evidence, inspect source/IR and simulator
regions as needed, propose a testable change, review all candidate paths, evaluate
the full contract, and feed the result back to the model. The host orchestrates rounds;
the Python core does not independently generate or schedule optimization.

Load the [HG-KDA skill](skills/hgkda/SKILL.md) in a model host, or use its equivalent
entry point: `python skills/hgkda/scripts/run_hgkda.py doctor`.

## Runtime and measurement

`python -m hgkda.runtime --help` exposes prepare, context, preflight, pair, simulate,
generate, evaluate, and trace actions. Configuration and boundaries are documented
in [public runtime](docs/runtime.md). The package works without `development/`.

Validated collection uses Ascend 910B and CANN 9.0.0 `msprof op` / `msprof op simulator`.
Newer MindStudio Ops Profiler documentation calls the standalone tool `msopprof`.
The tool invokes the vendor simulation library; formal performance comes from real
hardware. Strict candidates use Triton throughout; CANN/Ascend C and PyTorch references
are recorded separately. Full-call KGS timing, hardware profiling, and simulator costs
have separate scopes. Overlapping pipeline and instruction durations are not summed
into latency. Delivery status lives in the sibling `operators/index.json`.
