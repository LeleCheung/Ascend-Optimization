# KernelGenBench VCD native catalog

This v6.2 native catalog contains 2 operators and 20
replayable VCD cases.  Every correctness workload reads its exact VCD input
and golden output from `/data/dumps/kernelgenbench-vcd-peft-boft/<operator>/`;
timing reuses the same input
but executes the trusted reference on the allocated target device.

## Sources

- `peft`: KernelGenBench `908d0637832134a74274f03937d46aaee1533491` (2 operators).


The workload-to-KS3 mapping is recorded in `ks3_objects.json`.  The baseline
implementation copied into each operator's `assets/reference.py` remains under
its upstream license. It is self-contained and requires PyTorch plus a working
CUDA extension build toolchain; installing PEFT is not required.

## Correctness tolerances

KGS uses its dtype defaults unless a correctness workload contains an explicit
`tolerance` object.  This catalog has 20 workloads with
human-reviewed, dtype-specific tolerances.  Their auditable source of truth is
`tools/vcd_peft_boft_tolerance_policies.json`; the converter validates and
copies those policies into `correctness.jsonl`.  Timing workloads never carry a
correctness tolerance.  Every reviewed policy retains
`required_matched_ratio: 1.0`.

## Download data

From the repository root:

```bash
python3 tools/download_vcd_peft_boft_assets.py
```

The default destination is `/data/dumps/kernelgenbench-vcd-peft-boft`. Use
`--output-root` for a staging directory and `--operator` or `--limit` for a
smoke download. Existing files are checked and reused unless `--overwrite` is
supplied.

## Validate

```bash
python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-peft-boft/ops --delivery --mode standard \
  --output validation-vcd-native-static.json

python3 tools/validate_v62_native_catalog.py \
  data/kernelgenbench-vcd-peft-boft/ops --delivery --mode standard --runtime \
  --target-device cuda:0 --runtime-timeout 300 \
  --output validation-vcd-native-runtime.json
```

After starting KernelGen Server, submit the trusted oracle and its private
assets through the same candidate path used by a real solution:

```bash
python3 tools/run_vcd_reference_as_solution.py \
  --server http://127.0.0.1:8000 \
  --catalog data/kernelgenbench-vcd-peft-boft \
  --operator fast_block_diag_forward \
  --output reference-as-solution.json
```

This calls `/inspect`, optional `/preflight`, and `/evaluate`. Correctness reads
the configured `input_path` and `output_path`; timing reads `input_path` and
runs both reference and candidate on the Server device.

Correctness golden files can contain only Tensor leaves.  The oracle therefore
projects nested outputs to their Tensor leaves in VCD serialization order.
Operators whose source return contains non-Tensor structural leaves: none.
When a source operator has workload-dependent output arity, missing trailing
positions are represented by empty float32 Tensors.  This projection and
padding are part of the catalog's public return contract; no Tensor leaf from
the source correctness result is discarded.
