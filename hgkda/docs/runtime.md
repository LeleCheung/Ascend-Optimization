# Public runtime

Install `hgkda/`, then use `python -m hgkda` for offline analysis and
`python -m hgkda.runtime` for device/model actions. These interfaces are packaged;
they do not locate scripts inside a development checkout.

## Prepare and evaluate

Provide explicit environment JSON with `container`, `device`, `device_python`, `sdk`,
`lock`, `kgs_server`, shared run directories, package path, and device configuration.
The device needs matching Triton, torch_npu, and CANN. KGS is an external configured
service; public commands do not start unrelated infrastructure.

```sh
python -m hgkda.runtime prepare --operator-directory /path/operators/log10_ --package /path/hgkda --environment /path/environment.json --run-directory /path/runs/new-run
python -m hgkda.runtime context --config /path/runs/new-run/config.json
python -m hgkda.runtime preflight --config /path/runs/new-run/config.json
python -m hgkda.runtime evaluate --config /path/runs/new-run/config.json
```

`prepare` copies the selected delivery source exactly and marks
`prior_best_source_provided=true`. This is a prior-best regression start, not generation
from scratch. A relocated original contract retains its reference identity; a new
contract records a new hash. Reviews live in `source-reviews/<SHA>.json` and expire
when source changes. Preflight cannot replace full evaluation.

## Collection and generation

```sh
python -m hgkda.runtime pair --config /path/runs/new-run/config.json --case log10_-timing-3
python -m hgkda.runtime simulate --help
python -m hgkda.runtime generate --config /path/runs/new-run/config.json --invocation iteration-1
```

The current model adapter invokes local Claude Code. An external `model-settings.json`
selects the model and credentials; never commit this file into research artifacts.
Each invocation requests a concrete change. The host handles review, rounds, and
stopping; do not describe this as autonomous scheduler behavior.

Default `optimization_policy=performance_first` imposes no search budget and does not
stop at the acceptance threshold. `generate` defaults to `--max-turns 0` and
`--generation-timeout 0` (unlimited within the invocation). No outer generation deadline
is added. The host returns full results and continues after failures unless the user
sets a budget or stop. `optimization_policy=target_only` implements an explicit stop-at-target
request. Model context capacity and device-service timeouts still apply. Runtime
transport timeouts support recovery and do not become optimization budgets.

Paired collection preserves every device task and CSV when a call launches multiple
kernels. Comparisons retain kernel distributions, task counts, totals, and names for
that capture. One task duration is not a multi-kernel call latency. Logical CSV identity
ignores timestamp-only filename changes while preserving raw SHA. Full scores come from KGS.

Select multi-kernel simulation with
`python -m hgkda.runtime.collect_simulator --kernel-name <discovered-symbol>` using
current-source discovery and invocation data; record why that kernel was chosen.
Missing captures remain explicit missing stages. Do not fix targets by historical filenames.

Trace units are cross-checked against non-synchronization instruction CSVs on the same
PC/core. Scalar records may have different scopes; do not force a match. Preserve
piecewise clocks when frequency changes. Synchronization interpretation is described
in the [skill interface](../skills/hgkda/references/interface.md).

External device imports use explicit `device_pythonpath`; the runtime does not default
to an internal FlagGems snapshot.

## Decisions and replay

Actions automatically record events and source/artifact bindings. Before changing a
candidate, submit `python -m hgkda.runtime decision`; export with
`python -m hgkda.runtime trace`. See [continuous trace](../skills/hgkda/references/trace.md).
Original agent logs remain immutable. Public packaged knowledge retains historical
provenance identifiers without depending on their original files at runtime.

## Acceptance layers

| Layer | Entry point | Scope |
| --- | --- | --- |
| Full KGS contract | `runtime evaluate` -> external KGS | Frozen correctness, special values, and full-call timing |
| HG-KDA supplementary | Same evaluate action -> bound `operators/<name>/checks/` | Dual references, source paths, special values, required semantics |
| Native FlagGems | `runtime upstream-test` -> external pinned workspace | Import/dispatch, native accuracy, native benchmark |

`validation.py` interprets results rather than executing every test. Bind each layer
to the same integrated source; rerun after changes. Native benchmark scores do not
replace KGS scores. Software unit tests in `development/tests` are separate.

## Native upstream acceptance

`python -m hgkda.runtime upstream-test --config /path/upstream-test.json` verifies an
explicit external Git workspace/commit, source hashes, actual imports, device, and
new output directory. The generating model owns integration code. Tests and tolerances
stay unchanged. The adapter selects exact operators via upstream `tools/run_tests.py`,
does not enable quick mode, and preserves raw results. Missing timing, collection
errors, or all-skipped cases cannot pass. See [upstream validation](../skills/hgkda/references/upstream-validation.md).
