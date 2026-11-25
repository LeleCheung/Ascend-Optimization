Load skills/hgkda/SKILL.md and read task.json, context-brief.json, and performance-feedback.json.

Restore compatibility of an existing strict Triton delivery before performance work.
Current formal evaluation failed because the generated launcher received Python integer
pointers and called Tensor.size(). PyTorch/CANN checks passed; preserve math and tiling.

Inspect cached launcher calls in candidate.py and supply actual current-call Tensor
objects where the current ABI requires them. Preserve device kernels, scalar argument
order, tiling, precision, cache keys, current-stream queries, and single cold execution.
Do not disguise pointers as tensors, change compute by evaluation environment, or use
other backends. Read supplied runtime API excerpts without probing host devices.

Use only Edit on candidate.py/diagnosis.json. Register input SHA, failure evidence,
compatibility hypothesis, expectation, and validation before edits. You generate the
candidate; Codex reviews/schedules. Call bridge preflight after the change; at
awaiting_source_review preserve diagnosis and hand off without retry loops. No probe
scripts or contract/tool/knowledge edits. Repair only this mechanism, then continue
performance recovery after full results.
