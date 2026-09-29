# Validation record

- Source: KernelGenBench vLLM 0.28 CUDA
- Revision: `aae6c21ead635874713fc65d0d7c589ed774131c`
- Runtime: vLLM 0.28.0, PyTorch 2.13.0+cu130, Triton 3.6.0, CUTLASS DSL 4.6.2
- VCD-passing operators included: 18
- Correctness workloads: 138
- Timing workloads: 138
- Correctness policy: every workload keeps `required_matched_ratio=1.0`
- Hardware used for live KGS reference-as-solution validation: NVIDIA H20-3e
- Live result: 18/18 operators and
  276/276 correctness/timing workloads passed

The candidate submitted during validation was the trusted reference adapter,
so timing proves path viability and is not an optimization claim.
