# Candidate and reference contract

Strict candidates use Triton for every device computation, copy, reduction, scan,
layout conversion, and fallback. CANN libraries and identified Ascend C implementations
are references. Python/C++ handles allocation, metadata, arguments, compilation caches,
state, and launching only.

Preserve dtype, stride, offset, special values, in-place/out identity, current device,
and stream semantics. Cache immutable metadata only, never tensor inputs/results or
old streams. Do not hide host fallback or change tests to fit the candidate.

Static screening identifies known risks; it does not establish whole-path compliance.
Bind a source review, full KGS correctness/performance evaluation, and required supplementary
checks to the candidate. Default to one formal evaluation; explain repeats. Historical
mixed-backend scores are not strict Triton results.
