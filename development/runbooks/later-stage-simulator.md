# Simulating later stages of multi-kernel calls

When earlier stages consume a capture timeout, snapshot hardware-produced intermediate
inputs and replay the target launch alone. This supports regions/timing diagnosis;
full correctness, performance, and candidate retention still use the original contract.

1. Finish full KGS and required checks for original/reviewed IR. Save source, IR,
   actual binaries, and paired profiling.
2. Under the shared device lock run `python -m hgkda.runtime.launch_replay --spec
   <paired-spec.json> --kernel-name <symbol> --output <snapshot>`. Use actual device
   runtime; preceding kernels execute on hardware, then synchronize/snapshot before target.
3. Use an independently versioned observation package with `hgkda.runtime.collect_simulator`,
   adding `--kernel-replay <snapshot>/manifest.json`. Kernel, discovery, contract,
   case, and source bindings must match.
4. Verify source/tensor SHA, other-stage IR, compiler receipts, and target binary.
   Analysis aliases still require byte-identical hardware binaries.
5. Save hotspot regions and execution/wait/idle windows. Partial timeout captures retain
   exit/boundaries. Record per-PC clocks, source mapping, and completion separately.

The snapshot supports contiguous tensors and preserves identical views sharing storage;
other alias views are rejected explicitly. Preserve target ABI, arguments, grid, and
original Triton function. Snapshots are diagnostic inputs, not substitutes in evaluation.

GQA uses `attention_profile_driver.py --snapshot-kernel _gqa_fwd_kernel`, followed by
`--kernel-replay <output>/launch-snapshot/manifest.json`. Preserve contract input generation,
original arguments, preceding launches, and tensor identity. Source and IR-override modes
have separate checks; IR mode fixes all preceding-stage IR. `gqa_continue_observe.py`
collects auxiliary profiling before snapshot/replay. Formal full calls include every
stage and allocation.

Direct paired-driver scripts load sibling helpers explicitly. Preserve failed captures,
old packages, and snapshots. Internal coordination: `development/tools/research/ir_isolate_collection.py`,
`ir_retry_simulator.py`, and `ir_collect.py`. Scatter uses
`--kernel-name _scatter_add_sort_out_kernel --isolate-target` after paired profiling.
Verify observation-package support before creating output. Historical successful
package: `runs/ir-performance-20261119/observation-method/hgkda-v5`; older ir_collect_v3.py
retains historical identity.

[Verified scatter example](../reports/ir-performance-20261119/scatter-collection-recovery.md)
records binary identity and 152/152 PC clock matches for a partial sort_out window.
Capture validation and later candidate performance acceptance remain distinct.
