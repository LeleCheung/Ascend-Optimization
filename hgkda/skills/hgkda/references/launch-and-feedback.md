# Launch cost and performance feedback

Similar full-call times across small inputs, despite increasing large-input device
work, suggest separating host dispatch and device execution. Report full-call,
asynchronous submission, device-event span, and task time separately. Event spans
may contain host issue gaps and are not automatically kernel-only time.

## Triton launch path

Ordinary `kernel[grid](tensor, ...)` performs argument binding, device/stream handling,
and dispatch. Inspect whether compiled results, packed metadata, function handles,
grids, and host templates can be cached and used with the actual generated launcher.
Device work must still execute that Triton artifact.

Inspect the current launcher ABI rather than guessing from other backends. Validate
dtype, shape, stride, alignment, device, and current stream every call. Never cache
input/output values or old streams. Keep verified ordinary-Triton fallbacks for
misalignment, tails, noncontiguous layout, and debug/profiling modes. In-place cold
starts execute once, without retrying an operation that may already have run.

Direct launchers may retain Python slots for explicit `tl.constexpr` arguments.
The observed `make_launcher` maps these to `O`; verify added compile-time argument
slots rather than copying another kernel's count. Do not guess ABI positions or
catch/reissue failed in-place launches.

Historical SELU launch caching reached about 1.039x CANN, while an independent ordinary
launch start achieved 0.210x. Subsequent launch/queue/tiling work reached 0.887x, with
36/36 formal, 42/42 checks on each side, and 33/33 targeted checks. The faster historical
source remains retained. These implementation-level observations do not establish
single-variable causality. Use `knowledge --operator selu_` for bound method evidence;
do not present copied historical source as newly generated model output.

## Continue using feedback

1. Match full formal results to fixed case names, dtype, shape, and timing scope.
2. Rank losses by `-log(speedup)`, covering small and largest inputs rather than only
   the most expensive absolute timings.
3. Read source, actual IR, and matching captures; select one mechanism. Record empty
   knowledge retrieval and investigate transferable methods when no scoped record exists.
4. Review new source, run required supplementary checks and one full evaluation, then
   return full feedback. Correctness alone does not satisfy performance stop conditions.
5. Compare same-contract source versions and retain the best. Preserve failures without
   overwriting deliveries or selecting the highest same-source repeat.
