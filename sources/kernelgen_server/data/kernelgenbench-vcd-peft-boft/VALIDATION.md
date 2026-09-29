# Validation record

## Source

- KernelGenBench PR: https://github.com/lukatao/KernelGenBench/pull/5
- Source revision: `908d0637832134a74274f03937d46aaee1533491`
- Operators: `fast_block_diag_forward`, `fast_block_diag_backward`
- Cases: 10 per operator, covering float16 and float32

## VCD reference-as-solution

The VCD run used the PR baseline as both the trusted reference and submitted
solution on an NVIDIA H20-3e with PyTorch `2.11.0+cu129` and CUDA 12.9.

| Operator | Correctness compare | Reference timing | Solution timing |
| --- | ---: | ---: | ---: |
| `fast_block_diag_forward` | 10/10 passed | 10/10 produced | 10/10 produced |
| `fast_block_diag_backward` | 10/10 passed | 10/10 produced | 10/10 produced |

VCD therefore executed both copies of the CUDA implementation, compared their
outputs, and entered timing only after correctness passed. The 20 input and 20
reference-output objects were subsequently found in KS3.

## KGS safetensors replay

The source-specific downloader fetched all 40 files into
`/data/dumps/kernelgenbench-vcd-peft-boft`. KGS then read those fixed inputs and
Golden outputs while the trusted `assets/reference.py` was submitted through
the normal candidate path.

- Hardware: NVIDIA H20-3e, isolated GPU 7
- Runtime: PyTorch `2.9.0+cu128`, Triton `3.5.0`
- KGS: v6.3.1 / API v6.2, commit `a02cb38`
- `fast_block_diag_forward`: preflight passed; 10/10 correctness and 10/10
  timing workloads passed; reference-as-solution geometric mean `1.2889856`
- `fast_block_diag_backward`: preflight passed; 10/10 correctness and 10/10
  timing workloads passed; reference-as-solution geometric mean `0.9998856`

The speed ratios only validate the execution path; they are not optimization
claims because the candidate and reference contain the same trusted kernel.
Every correctness workload retains `required_matched_ratio=1.0`.
