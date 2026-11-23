# Measurement rules that are easy to get wrong

Every rule here was learned by getting it wrong first. The cost is noted where it was
paid, because a rule with a price attached is harder to talk yourself out of.

## Never parallelise `hexagon-sim`

**One simulator at a time.** `rung0 grade` defaults to `--jobs 1` and *refuses* more
rather than clamping.

Parallel simulation starved the machine, left orphaned `hexagon-sim.exe` processes
that survived every kill, and — the part that actually matters — made a
contention-induced timeout indistinguishable from a wrong answer. **352 verdicts had
to be discarded and re-taken** over exactly this.

Budget wall-clock instead. A T2 attempt is ~4 minutes serially, against seconds for a
T0. Parallelism belongs *across batches*, never inside the simulator.

A related accounting fix: 19 attempts that timed out in the simulator were initially
booked as wrong answers. Reclassifying them as ungraded moved measured correctness
from 95.3% to 98.6%. A timeout is not a wrong answer, and a harness that cannot tell
them apart is reporting a number about itself.

## Compare `kernel_cycles` to `expert_kernel_cycles`

Never whole-program `cycles`. Never host wall clock.

Harness and CRT overhead is roughly constant (~155–190k cycles), so a whole-program
ratio scales inversely with kernel size. On a small kernel it looks like a simulator
problem; it is arithmetic.

## Say which detector produced a VTCM number

`used_vtcm` exists twice and the two do not always agree:

- **Static** (`anticheat._disasm_has_vtcm`) reads the disassembly and needs no timing.
  This is what `hexkernels/forge/verify.py` uses.
- **Runtime** is a PMU counter. It needs timing enabled and **silently reads 0
  otherwise** — indistinguishable from a kernel that genuinely did not use VTCM.

The asymmetry is a disclosure item, not an implementation detail. See
[ANTI-CHEAT.md](ANTI-CHEAT.md).

## Fleet width 16, not the probe's recommendation

Wider oversubscribes and — measured — flips some kernels to *incorrect*. The probe
optimises throughput, not fidelity. A faster run that changes answers is not a faster
run.

## Pin the toolchain version for any cycle comparison

Codegen differences across releases account for known outliers. A cycle count without
a toolchain version attached is not comparable to anything.

## On-chip measurement: committed packets, not cycles

The DSP PMU is readable from an unsigned PD. Of what it reports:

- **committed packets is deterministic** — the same work gives the same count;
- **cycles swings about 21%** run to run.

So on-device comparisons should be made on packets. Cycles on silicon carry real
variance that a single run will not show you.

`cycles_total=0` is the expected shape of failure on silicon, not an edge case: PCYCLE
returns 0 when `SYSCFG.PCYCLEEN` is clear, and a user-mode unsigned PD cannot set it.
`hexkernels/device/qdc/verdict.py` refuses a job whose only measurement is zero —
[QDC.md](QDC.md) explains why that check exists.

## Instruction counts depend on the ELF filename

`insns` varies with the name of the ELF being measured. Keep the filename fixed across
a comparison, or the difference you are reading is the path length.

## `witness_build/` is a cache, not a result

It regenerates in roughly 14 seconds per batch. Only `results.json` holds real
simulator time. Never treat a regenerated witness tree as evidence of a measurement —
it is evidence of a build.
