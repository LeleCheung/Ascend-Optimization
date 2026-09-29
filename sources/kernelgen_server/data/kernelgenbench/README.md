# KernelGenBench v6.2 native catalog

This catalog is deterministically converted from the checked-in KernelGenBench v5.1 210-operator catalog by `tools/convert_kernelgenbench_v5.py`. It retains the complete 57,603 correctness and 28,198 timing workloads as `correctness_full.jsonl` and `timing_full.jsonl`, and selects at most 200 active workloads per operator and phase with `input-value-pairwise-greedy-v1`. It moves each embedded reference into `ops/<group>/<name>/oracle.py`, preserves the original `pointwise`, `vllm`, and `cublas` groups, and exposes a pure v6.2 public ABI in `definition.json`.

KGS evaluates only `correctness.jsonl` and `timing.jsonl` (15,353 and 11,695 workloads). The `_full` files are inactive archives and are the authoritative inputs for deterministic resampling:

```bash
PYTHONPATH=. python3 tools/sample_v62_catalog_workloads.py \
  data/kernelgenbench --limit 200
```

The original v5.1 source remains recoverable from Git commit `4a5769a` and its descendants before this migration. Regeneration must use a clean checkout of that source catalog and run the tool with `PYTHONPATH=.` from the KernelGen Server repository root.
