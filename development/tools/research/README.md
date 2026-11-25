# Research orchestration

These tools support explicit source-frozen compiler/evaluation studies. They are not
part of the installed runtime and do not authorize restarting completed campaigns.

- `ir_capture.py`, `ir_lowering_review.py`, `ir_binary_compare.py`: compiler artifacts,
  lowering review, exact binary identities, and independent timing comparisons.
- `ir_model.py`, `gqa_model_start.py`, `gqa_repair_start.py`: bound model inputs and
  registered pre-change decisions; candidate edits belong to CC/DeepSeek.
- `ir_campaign.py`, `ir_collect.py`, `ir_isolate_collection.py`: explicit Linux/NPU
  execution with exclusive locks, full results, and separately scoped captures.
- `gqa_review_bindings.py`, `model_result_review.py`: actual read/decision evidence
  and read-only result handoff.

Device scripts need the configured CANN/Triton environment. Importing offline identity
helpers must not require Linux device locking. Run outputs, model settings, and
credentials belong outside the checkout. Reports retain failed and unretained branches.

Dated presentation builders and fixed-report generators moved to
[the presentation archive](../../presentation_archive/README.md). Follow the relocation
manifest for old commands; do not rewrite frozen reports to change their historical paths.
