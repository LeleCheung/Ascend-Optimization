# Continuous trace and pre-change decisions

One round is one named model invocation. Count model calls, edits, preflights, and
formal evaluations separately. Public runtime records these without requiring
`development/` or changing candidate computation/contracts.

## Automatic records

`generate` saves before/after source, model logs, initial context, method identity,
and diagnosis references. `preflight`/`evaluate` save start/end events, current source,
checks, and formal results. `context`/`pair`/`simulate` also emit events. Specialized
probes retain their own artifacts and are referenced explicitly. `returned` means
function return, not correctness, completeness, or performance acceptance.

Source snapshots deduplicate by SHA while events remain distinct, including failures,
unchanged calls, and repeated evaluations. Mutable workspace artifacts are content
snapshots. Raw `agent-events.jsonl` retains intermediate edits/reads without automatically
promoting each read into decisive evidence.

```text
<run>/
  trace/
    events/<event-id>.json          immutable action/decision events
    sources/<SHA>.py               input/output/pre-decision source snapshots
    artifacts/<SHA>.json           mutable context/feedback snapshots
    evidence/<SHA>.<ext>           immutable cited evidence
    decisions/<decision-id>.json   pre-change evidence, hypothesis, expectation
    rounds/<invocation>/round.json model-call summary and references
  agent-calls/<invocation>/        original model logs and method identities
  attempts/<SHA>/                  device preflight originals
  formal-<SHA-prefix>/             KGS and supplementary originals
```

Start/end share event IDs. Explicit `trace_parent_round_id` selects ancestry; otherwise
parent remains unknown, not inferred from equal SHA or timestamps. Killed processes
may leave unclosed start events. Appending model logs bind `prefix_bytes` and prefix
SHA. Legacy logs without lengths can be recovered only by an existing exact-SHA prefix;
do not replace an old digest with a new one. Snapshot mutable source-provenance files.

## Register before editing

Write workspace `diagnosis.json`, then run:

```sh
python -m hgkda.runtime decision --config /path/run/workspace/agent-config.json --record diagnosis.json
```

The CC bridge provides the same action. Generation writes the invocation ID into its
configuration. Source must still be the pre-change version when registering; small
edits for one mechanism need not repeat a long analysis each time.

```json
{
  "source_sha256": "current-source-sha256",
  "evidence": [{
    "path": "simulator-before/core0.veccore0-regions.json",
    "category": "simulator", "role": "disambiguation",
    "finding": "A regional wait observation distinguishes two proposed mechanisms"
  }],
  "hypothesis": "Current mechanism",
  "alternatives": ["Alternative explanation"],
  "discriminating_observation": "Observation separating the alternatives, or empty",
  "change": "Proposed mechanism change",
  "expected": "Observable change and direction",
  "validation": "Recollection and full evaluation plan"
}
```

Categories: `paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other`.
Roles: `trigger/disambiguation/mechanism/background`. Full per-case scores are evaluation,
not paired hardware profiling. Cite only tools actually used and only real disambiguation.
Cross-version evidence uses `background` and its original source SHA. Current failures
can trigger current edits; old feedback is not current-source measurement.

Paths are run-relative (`workspace/<file>` for workspace files) or explicit absolute
paths. Registration computes SHA and copies cited bytes into `trace/evidence`, retaining
original identity. Reject structured source-binding mismatches. For an artifact from
another source, correct its role/provenance rather than modifying evidence or retrying
unchanged registration. Register a short valid decision, then edit promptly.

Records preserve the author's declared basis, not proof of independent causality.
Store post-change diagnosis/metrics/results separately and compare predictions.
Historical missing decisions remain `not_recorded`.

## Export

```sh
python -m hgkda.runtime trace --config /path/run/config.json --output /path/run/trace-export-01.jsonl
```

Use a new output file. Exports sort by recorded time with event/source/artifact identity;
raw logs retain internal tool ordering. Clock order alone does not establish parallel
causality. Offline tests cover identities, failures, duplicate source, and CLI behavior;
new device loops still need their own validation. Do not re-register frozen history or
describe logging support as a complete autonomous optimizer.
