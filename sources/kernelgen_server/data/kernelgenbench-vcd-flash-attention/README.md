# FlashAttention VCD native catalog

This KGS v6.2 native catalog contains 25 operators and
604 correctness plus 604 timing workloads. Inputs and
Golden outputs are the exact artifacts produced by VCD and recorded in
`ks3_objects.json`.

## Download workload tensors

From the repository root:

```bash
python3 tools/download_vcd_flash_attention_assets.py
```

The default destination is `/data/dumps`. Use `--output-root`, `--operator`,
or `--limit` when a different destination or subset is required. The script
downloads only this catalog's FlashAttention objects.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-flash-attention/ops --delivery --mode standard
```

Source revision: `4bd58713c9bcab745c7b875d776216c0d883398a`. Runtime: KernelGenBench FlashAttention environment.
