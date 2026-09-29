# Flash Linear Attention VCD native catalog

This KGS v6.2 native catalog contains 107 operators and
642 correctness plus 642 timing workloads. Inputs and
Golden outputs are the exact artifacts produced by VCD and recorded in
`ks3_objects.json`.

## Download workload tensors

From the repository root:

```bash
python3 tools/download_vcd_fla_assets.py
```

The default destination is `/data/dumps`. Use `--output-root`, `--operator`,
or `--limit` when a different destination or subset is required. The script
downloads only this catalog's Flash Linear Attention objects.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-flash-linear-attention/ops --delivery --mode standard
```

Source revision: `b87ac6ab5607bfb8ea88c30e444edb31ecd0a054`. Runtime: KernelGenBench FLA environment.
