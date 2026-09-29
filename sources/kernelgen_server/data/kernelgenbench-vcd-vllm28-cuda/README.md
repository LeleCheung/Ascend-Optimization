# vLLM 0.28 CUDA VCD native catalog

This KGS v6.2 native catalog contains 18 operators and
138 correctness plus 138 timing workloads. Inputs and
Golden outputs are the exact artifacts produced by VCD and recorded in
`ks3_objects.json`.

## Download workload tensors

From the repository root:

```bash
python3 tools/download_vcd_vllm_cuda_assets.py
```

The default destination is `/data/dumps`. Use `--output-root`, `--operator`,
or `--limit` when a different destination or subset is required. The script
downloads only this catalog's vLLM 0.28 CUDA objects.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-vllm28-cuda/ops --delivery --mode standard
```

Source revision: `aae6c21ead635874713fc65d0d7c589ed774131c`. Runtime: vLLM 0.28.0, PyTorch 2.13.0+cu130, Triton 3.6.0, CUTLASS DSL 4.6.2.
