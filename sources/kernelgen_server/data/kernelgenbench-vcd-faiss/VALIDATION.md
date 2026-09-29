# Validation record

- Source: KernelGenBench Faiss
- Revision: `fa75616e431da55898edb3ce7eff828e98d94214`
- Runtime: PyTorch 2.12.0+cu130, Triton 3.7.0, faiss-cpu 1.14.3
- VCD-passing operators included: 15
- Correctness workloads: 85
- Timing workloads: 85
- Correctness policy: every workload keeps `required_matched_ratio=1.0`
- Hardware used for live KGS reference-as-solution validation: NVIDIA H20-3e
- Live result: 15/15 operators and
  170/170 correctness/timing workloads passed

The candidate submitted during validation was the trusted reference adapter,
so timing proves path viability and is not an optimization claim.
