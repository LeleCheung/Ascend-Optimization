# Metrics and optimization hypotheses

These thresholds are versioned heuristics, not learned universal laws.

- MTE activity >= 0.7: investigate contiguity, movement granularity, repeated traffic,
  tiling, and compute overlap.
- Vector/Cube activity >= 0.7: inspect actual instructions, conversions, reductions,
  effective work, and pipeline organization.
- Scalar activity >= 0.5: inspect control, addresses, dynamic loops, and program coverage.

When several pipelines are active, inspect the execution path, IR, and instructions.
Crossing a threshold prioritizes investigation; it does not establish the latency bottleneck.
A lower value does not rule one out.

Knowledge records include operator type, shape/precision, metric scope, action,
validation, failures, provenance, and source identity. New experience starts as draft;
validate and deduplicate it before promotion. Conflicting records retain their scope
and counterexamples rather than replacing failure evidence.
