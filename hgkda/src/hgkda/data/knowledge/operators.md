# Operator types and candidate experiments

- Pointwise/in-place: inspect launch paths, program coverage, address calculation,
  tails, and special values. Actual IR may reveal unrolling, broadcasting, or UB allocation.
- Reduction/scan: establish axes, transposes, strides, and associativity. Changes in
  accumulation order or precision need numerical checks beyond small inputs.
- Indexing/scatter/sampling: check boundaries, strides, repeated indices, write conflicts,
  atomics, initialization, and layout conversions across every candidate path.
- Matrix/fused kernels: verify the compute path, then inspect panel reuse, K tiling,
  buffer occupancy, synchronization, and tails. Source-level reuse need not reduce actual traffic.
- Copy: references may use DMA; candidates must use Triton. Inspect alignment, address
  calculation, and device/host dispatch, not just total latency.

Each change records evidence, mechanism, expected metric direction, and validation.
Targeted small experiments can distinguish mechanisms; they cannot replace complete
acceptance or hide poor performance on larger cases.
