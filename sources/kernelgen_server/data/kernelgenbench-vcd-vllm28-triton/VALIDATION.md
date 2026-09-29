# Validation record

- Source: KernelGenBench vLLM 0.28 Triton
- Revision: `c1d369a914558e9908789ac747a3440d605ba06e`
- Runtime: vLLM 0.28.0, PyTorch 2.13.0+cu130, Triton 3.7.1, CUTLASS DSL 4.6.2
- VCD-passing operators included: 41
- Correctness workloads: 328
- Timing workloads: 328
- Correctness policy: every workload keeps `required_matched_ratio=1.0`
- Hardware used for live KGS reference-as-solution validation: NVIDIA H20-3e
- Live result: 41/41 operators and
  656/656 correctness/timing workloads passed

The candidate submitted during validation was the trusted reference adapter,
so timing proves path viability and is not an optimization claim.
