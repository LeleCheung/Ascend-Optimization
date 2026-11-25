# Evidence examples

| Example | Purpose | Device required |
| --- | --- | --- |
| `paired-expm1/` | Paired CSV aggregation, source identity, model context | No |
| `matmul-instructions/` | Original simulator listing, hotspot ranking, source regions | No |
| `kgs-rad2deg/` | Reviewed strict Triton candidate and full CANN-reference request | Yes |

Samples retain original bytes, provenance, and hashes. The expm1 example demonstrates
diagnosis rather than an accepted CANN speedup. The matrix listing adds a verified
source binding; `original-listing.json` preserves the original instructions.

The KGS example contains a reviewed candidate, a full 36-case contract, and its
reference library. Initialize the environment and verify device ownership and service
status before evaluation. Use one request by default, not repeats selected for high scores.

```sh
hgkda kgs upload --server http://127.0.0.1:19661 --archive examples/kgs-rad2deg/operator.tar
hgkda kgs evaluate --server http://127.0.0.1:19661 --request examples/kgs-rad2deg/request.json --review examples/kgs-rad2deg/review.json --run-directory /path/out/new-evaluation
```

The candidate is also delivered at `operators/rad2deg_/candidate.py` relative to the
repository root. Server setup is covered by [optional backends](../docs/backends.md).
