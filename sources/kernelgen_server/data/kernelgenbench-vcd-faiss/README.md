# Faiss VCD native catalog

This KGS v6.2 native catalog contains 15 operators and
85 correctness plus 85 timing workloads. Inputs and
Golden outputs are the exact artifacts produced by VCD and recorded in
`ks3_objects.json`.

## Download workload tensors

From the repository root:

```bash
python3 tools/download_vcd_faiss_assets.py
```

The default destination is `/data/dumps`. Use `--output-root`, `--operator`,
or `--limit` when a different destination or subset is required. The script
downloads only this catalog's Faiss objects.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-faiss/ops --delivery --mode standard
```

Source revision: `fa75616e431da55898edb3ce7eff828e98d94214`. Runtime: PyTorch 2.12.0+cu130, Triton 3.7.0, faiss-cpu 1.14.3.
