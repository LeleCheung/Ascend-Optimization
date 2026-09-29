# Validation record

- Source: KernelGenBench Flash Linear Attention
- Revision: `b87ac6ab5607bfb8ea88c30e444edb31ecd0a054`
- Runtime: KernelGenBench FLA environment
- VCD-passing operators included: 107
- Correctness workloads: 642
- Timing workloads: 642
- Correctness policy: every workload keeps `required_matched_ratio=1.0`
- Hardware used for live KGS reference-as-solution validation: NVIDIA H20-3e
- Live result: 107/107 operators and
  1284/1284 correctness/timing workloads passed

The candidate submitted during validation was the trusted reference adapter,
so timing proves path viability and is not an optimization claim.
