# Native reference construction and qualification

These tools require matching Linux/CANN/PyTorch NPU environments; the public offline
core remains standard-library-only.

| Tool | Purpose |
| --- | --- |
| `build_cann_reference.py` | Rebuild the pinned example CANN host bridge |
| `../operators/expm1_/probe_cann.py` | Official legacy Expm1 precision-mode qualification |
| `../operators/uniform/` | Uniform+Clamp construction, qualification, and full evaluation |

[Operator tools](../operators/README.md) organize targeted checks. Earlier builders
that directly patched candidates are retired in [archive](../archive/README.md).

Reference device math uses vendor APIs; host bridges handle descriptors, allocation,
scalars, generator state, and streams. Strict candidate math/movement remains Triton.
Uniform+Clamp is an explicitly composed same-semantics reference: sample using original
from/to, then clamp to representable values inside the half-open interval. Include
necessary composition overhead and verify interval/statistics/layout/stream/state.
Qualification and full KGS results remain separate.

Use new output directories and exclusive project device locks. Manage only identified
batch services. Historical details: [regression record](../../reports/regression-20261109.md).
Cumulative-trapezoid mathematical contracts are separate: CPU PyTorch verifies intermediate
rounding/FP32 accumulation; formal references still execute all CANN calls on NPU.
Preserve unstable original-default reference evidence without claiming its acceptance.
