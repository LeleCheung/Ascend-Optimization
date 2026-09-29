# Validation record

- Source: KernelGenBench FlashAttention
- Revision: `4bd58713c9bcab745c7b875d776216c0d883398a`
- Runtime: KernelGenBench FlashAttention environment
- VCD-passing operators included: 25
- Correctness workloads: 604
- Timing workloads: 604
- Correctness policy: every workload keeps `required_matched_ratio=1.0`
- Hardware used for live KGS reference-as-solution validation: NVIDIA H20-3e
- Live result: 25/25 operators and
  1208/1208 correctness/timing workloads passed

The candidate submitted during validation was the trusted reference adapter,
so timing proves path viability and is not an optimization claim.
