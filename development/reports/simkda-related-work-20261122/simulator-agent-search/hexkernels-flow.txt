# The agentic flow: from a PyTorch operator to an accelerated kernel

`hexkernels/forge/` turns an operator into a task, and a task into a judged attempt.
The pipeline's design commitment is that **the scalar C it emits is the question, not
the answer**.

## The stages, and what each one proves

| | stage | proves |
|---|---|---|
| a | trace | module → primitive graph (**FX required**) |
| b | annotate | iterator types + indexing maps, derived |
| b2 | linalg | the compiler's own Linalg IR (**torch-mlir required**) |
| c | plan | which mechanisms the **size** entitles this kernel to |
| d | emit | portable scalar C++ — the question |
| e | golden | run the module on seeded inputs |
| f | harness | a `main()` that checks a candidate against those values |
| g | **verify** | the reference compiles, runs, and passes its own harness |
| h | prompt | the acceleration request, written to disk |

**Stage (g) is the gate.** If the emitted reference cannot pass the harness generated
beside it, the emitter and the golden disagree and nothing downstream means anything —
a candidate judged against a broken harness tells you about the harness. A kernel whose
reference fails is reported as failed and **no prompt is written for it**.

## Both IRs are required inputs, not add-ons

This pipeline constructs kernels from the Torch FX graph *and* Linalg IR. A kernel
built without either is not a product of this pipeline. The code path for a missing
Linalg is fail-*soft* so the failure is visible and attributed — `linalg_error` names
it, and the reference and harness still build so the batch stays diagnosable. That is
not there to make Linalg skippable. **A batch whose `REPORT.md` shows Linalg coverage
below 100% is not shippable.**

## Where the mechanisms come from

Stage (c) does not ask a model what the kernel should use. It derives the entitlement
from the working-set size — see [TIERS.md](TIERS.md). Each kernel's `spec.json` carries
the reasoning verbatim in `mechanism_reasons`:

```json
["hmx", "parallel loops over a shared reduction of products is the shape the HMX tile
  matmul implements, and the dtype (2 B) fits its int8/fp16 datapath"]
```

This is what makes a mechanism claim falsifiable. The prompt demands a specific
mechanism because the size entitles it, and the anti-cheat detector then checks the
binary for that exact mechanism.

## The turn loop, and a result worth knowing

`hexkernels/gym/` closes the loop: the model writes a kernel, a profiler names the
bottleneck, and it is told which mechanism is missing. Episodes end at outcome tier 3 —
correct **and** using the hardware.

The ladder study measured what each addition buys. It is not what you would guess:

| rung | the ask | correctness | reached a mechanism |
|---|---|---|---:|
| 0 | bare, unaided | high | 3 / 640 |
| 1 | + a turn loop reporting failures | **100% on T0/T1** | **0 / 640** |
| 2 | + the mechanism named explicitly | high | 20 / 640 in source, 2 confirmed in ELF |

**Rung 1 drove correctness to 100% and mechanism use to zero.** A feedback loop that
optimises the thing you measure will optimise away the thing you did not. That result
is only visible because the detector reads the binary rather than the cycle count.

At rung 2, the first model-written mechanism engagement in the study arrived through
the *retry* channel rather than the prompt — and the count is not seed-stable: it
ranges from 2 to 8 across seeds, a range of 6. Quote it with the range attached.

## Running it

```bash
# one batch through every stage
python -m hexkernels.forge.run_batch --batch 17 --out witness_build/batch17

# a ladder rung end to end
python -m hexkernels.forge.rung0 generate --model <model> --seeds 5 --jobs 8
python -m hexkernels.forge.rung0 facts                      # pre-trace once
python -m hexkernels.forge.rung0 grade  --model <model> --tier T0,T1
python -m hexkernels.forge.rung0 report --model <model> --seeds 5
```

`grade` is serial by design and refuses `--jobs > 1`. Read
[MEASUREMENT.md](MEASUREMENT.md) before changing that — it cost 352 discarded verdicts
to learn.

## A caution on sampling

The 5 seeds are **not** 5 independent samples, and `pass@k` saturates on correctness
almost immediately — which makes it a poor summary statistic here, because correctness
is not the thing in question. Report mechanism engagement directly, with its seed
range.
