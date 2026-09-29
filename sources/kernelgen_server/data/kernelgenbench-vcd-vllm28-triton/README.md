# vLLM 0.28 Triton VCD native catalog

This KGS v6.2 native catalog contains 41 operators and
328 correctness plus 328 timing workloads. Inputs and
Golden outputs are the exact artifacts produced by VCD and recorded in
`ks3_objects.json`.

## Download workload tensors

From the repository root:

```bash
python3 tools/download_vcd_vllm_triton_assets.py
```

The default destination is `/data/dumps`. Use `--output-root`, `--operator`,
or `--limit` when a different destination or subset is required. The script
downloads only this catalog's vLLM 0.28 Triton objects.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-vllm28-triton/ops --delivery --mode standard
```

Source revision: `c1d369a914558e9908789ac747a3440d605ba06e`. Runtime: vLLM 0.28.0, PyTorch 2.13.0+cu130, Triton 3.7.1, CUTLASS DSL 4.6.2.
