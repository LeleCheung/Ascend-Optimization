<div align="center">

# hexkernels

### 538 Hexagon NSP kernels that **actually use the accelerator**

Every entry carries HVX or HMX intrinsics, a scalar reference to check it against,<br>
and a harness to run both. Plus the machinery that proves it.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Release](https://img.shields.io/github/v/release/sriharshapy/hexkernels)](https://github.com/sriharshapy/hexkernels/releases)

[![kernels](https://img.shields.io/badge/kernels-538-6f42c1)](docs/LIBRARY.md)
[![HMX](https://img.shields.io/badge/reach%20HMX-73-e05d44)](docs/LIBRARY.md)
[![speedup](https://img.shields.io/badge/median%20speedup-3.39%C3%97-success)](docs/LIBRARY.md)
[![no deps](https://img.shields.io/badge/runtime%20deps-none-brightgreen)](pyproject.toml)

[**Quickstart**](#quickstart) · [**The library**](#the-library) · [**Anti-cheat**](#1-anti-cheat--the-reason-this-exists) · [**Docs**](docs/) · [**Contributing**](CONTRIBUTING.md)

</div>

---

> [!IMPORTANT]
> **A scalar C file is never a library entry here.** It ships as `reference.c`, which is
> a different job. If the source has no vector or matrix intrinsic, it is not a kernel —
> and that rule is enforced by a test, not by taste.

## Quickstart

```bash
pip install -e .          # no SDK, no hardware, no third-party dependencies
```

```python
from hexkernels.library import find

find(hmx=True)                       # 73 kernels that reach the matrix engine
find(origin="expert", dtype="fp16")  # hand-written fp16, with measured speedups
find(tier="T2", buildable=True)      # complete bundles at one working-set tier
find(elf_confirmed=True)             # the disassembler confirms it, not just the source

k = find(hmx=True)[0]
k.source         # the accelerated kernel
k.reference      # scalar ground truth
k.speedup        # measured, against that reference
k.elf_confirmed  # True / False / None — None means never scanned, not "no"
```

Browsing and querying the corpus needs **nothing but Python**. You only need the
Hexagon SDK to build, simulate, or disassemble.

## The library

<table>
<tr><th>origin</th><th align="right">n</th><th>what it attests</th></tr>
<tr>
<td><code>expert</code></td><td align="right"><b>350</b></td>
<td>Hand-written, each measured against its own scalar baseline.<br>
Median <b>3.39×</b> · p90 <b>18.2×</b> · best <b>124.7×</b>. Ships 421 near-miss kernels.</td>
</tr>
<tr>
<td><code>mined</code></td><td align="right"><b>171</b></td>
<td>Written against operators mined from the PyTorch registry.<br>
⚠️ <b>126 carry a tier mismatch</b> — recorded in <code>tier_match</code>, not hidden.</td>
</tr>
<tr>
<td><code>model</code></td><td align="right"><b>17</b></td>
<td>Written by a language model. The survivors of 1,920 attempts.</td>
</tr>
</table>

**73 reach HMX**, the matrix engine. Spread across fp32 (168), int8 (96), fp16 (36),
uint8 (29), int16 (18) and mixed-precision variants.

<details>
<summary><b>Every entry has the same shape, whatever its origin</b></summary>

```
kernels/<origin>/<name>/
├── kernel.c       the accelerated kernel        ← the library entry
├── reference.c    scalar ground truth
├── harness.c      correctness + cycle harness
├── kernel_api.h   entry-point declaration        (expert only)
├── nearmiss_*.c   plausible WRONG kernels the harness must reject
├── PROMPT.md      the ask that produced it
└── spec.json      normalised metadata
```

So a consumer never branches on `origin`. The entry point is always
`candidate_kernel`. `kernels/index.json` is the whole corpus in one file;
`tools/assemble_library.py` rebuilds it.

`spec.json.bundle` answers "can I build this?" in one field:

| bundle | n | meaning |
|---|---:|---|
| `complete` | 467 | reference and harness both present |
| `harness-regenerable` | 71 | harness too large to ship — `HARNESS.md` has the command |

A harness embeds base64 golden vectors, so its size tracks the **working set**, not
complexity. The largest is 93 MB; those 71 alone would be 675 MB of an otherwise
16 MB corpus. They regenerate in ~14 s per batch. Nothing irreproducible was dropped.

</details>

## The three things worth reading

### 1. Anti-cheat — the reason this exists

> A working scalar loop passes a correctness test and often **beats** a mediocre
> vectorised kernel on cycles.

So a score based on speed does not merely fail to reward accelerator use — it
*actively punishes* it. Correctness and speed both fail as evidence. The only thing
that settles it is the disassembled ELF.

**This is not hypothetical.** All 17 model-written kernels have HVX intrinsics *in the
source*. Of the 12 that were scanned:

| ELF verdict | n |
|---|---:|
| ✅ confirms the mechanism | **9** |
| ❌ **refutes** it | **3** |
| ⬜ never scanned | 5 |

Writing the intrinsic is not the same as the binary containing it. →
[**`docs/ANTI-CHEAT.md`**](docs/ANTI-CHEAT.md)

### 2. The agentic flow — `hexkernels/forge/`

```
PyTorch op → FX graph → Linalg IR → MLIR fusion → affine loops → scalar C
                                                                     │
                                           the QUESTION, not the answer
                                                                     ▼
                         a model reconstructs it as accelerated Hexagon C
                                                                     │
                             judged against a golden harness, then the ELF
```

The pipeline refuses to skip a stage, and stage (g) is the gate: if the emitted
reference cannot pass the harness generated beside it, no prompt is written at all.
`hexkernels/gym/` closes the loop — a profiler names the bottleneck and the model is
told which mechanism is missing. →
[**`docs/AGENTIC-FLOW.md`**](docs/AGENTIC-FLOW.md)

### 3. Simulator and silicon — `core/`, `device/qdc/`

`core/` holds the toolchain wrapper, target detection, the simulator fleet and the
reward. `device/qdc/` runs the same kernels on real hardware — an adb-server **port
forward, not a shell**, which bills a session for its full timeout if you don't
release it.

`verdict.py` is vendored in full because it is what keeps a false pass off the money
path: *a job that ran zero tests once reported passing*, and `cycles_total=0` satisfied
a check that only tested for the substring. →
[**`docs/QDC.md`**](docs/QDC.md)

## Two ladders, both called T0–T3

> [!WARNING]
> `T0`–`T3` names **two unrelated things**. They have been confused for each other
> before. Always say which ladder you mean.

| | **outcome ladder** (per attempt) | **task tiers** (per task, by size) |
|---|---|---|
| **T0** | doesn't compile | fits L1D (≤16 KB) — `hvx` only |
| **T1** | compiles, wrong | up to L2 (≤1 MB) — `+ l2fetch` |
| **T2** | correct, but **scalar** | inside VTCM — `+ dma + vtcm` |
| **T3** | correct **and** uses the mechanism | exceeds VTCM — must stream tiles |

The mechanisms a task is entitled to are **derived from its size** — the corpus's
central claim. → [**`docs/TIERS.md`**](docs/TIERS.md)

## The study behind it

Can a model be got to use the accelerator at all? Three rungs, 1,920 attempts.

| rung | the ask | reached a mechanism |
|---|---|---:|
| **0** | bare, unaided | 3 / 640 |
| **1** | + a turn loop reporting failures | **0 / 640** |
| **2** | + the mechanism named explicitly | 20 / 640 in source · 2 confirmed in ELF |

**Rung 1 drove correctness to 100% and mechanism use to zero.** A feedback loop that
optimises what you measure will optimise away what you didn't. That is only visible
because the detector reads the binary. → [**`docs/study/`**](docs/study/)

## Documentation

| | |
|---|---|
| [**LIBRARY.md**](docs/LIBRARY.md) | The corpus: schema, the three origins, what each attests |
| [**ANTI-CHEAT.md**](docs/ANTI-CHEAT.md) | The detectors, and the `used_vtcm` asymmetry you must disclose |
| [**AGENTIC-FLOW.md**](docs/AGENTIC-FLOW.md) | The pipeline, stage by stage |
| [**QDC.md**](docs/QDC.md) | On-silicon measurement, and why completion isn't a verdict |
| [**TIERS.md**](docs/TIERS.md) | The two ladders, and the entitlement gate |
| [**MEASUREMENT.md**](docs/MEASUREMENT.md) | ⚠️ The traps. Read before quoting any cycle number |

> [!CAUTION]
> **Never parallelise `hexagon-sim`.** It starves the machine, orphans processes, and
> makes a contention timeout indistinguishable from a wrong answer — 352 verdicts had
> to be discarded and re-taken over exactly that.

## Layout

```
kernels/              the library — one dir per kernel, one schema, + index.json
hexkernels/
├── library/          load and query the corpus
├── anticheat/        the ELF disassembly detectors
├── forge/            the agentic pipeline (+ frontend: trace, graph, emit, schedule)
├── gym/              profile-and-tune loop
├── core/             toolchain, target detection, simulator fleet, reward
└── device/qdc/       on-silicon measurement
benchmark/            the frozen corpus definition
env/harness/          harness headers every build includes
tools/                the script that assembles kernels/
```

## Contributing

Kernels especially — but also detectors, docs, and anything that makes a claim here
more falsifiable. Start with [**CONTRIBUTING.md**](CONTRIBUTING.md); there are issue
templates for [contributing a kernel](https://github.com/sriharshapy/hexkernels/issues/new?template=kernel_contribution.yml)
and for [disputing a verdict](https://github.com/sriharshapy/hexkernels/issues/new?template=false_verdict.yml).

A wrong verdict is the most serious kind of bug here — more serious than a crash.
Correcting a result is always welcome, including this project's own.

## License

[Apache-2.0](LICENSE). See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community
expectations.
