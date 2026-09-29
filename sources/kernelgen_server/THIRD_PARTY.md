# Third-party provenance

KernelGen Server is licensed under Apache-2.0.

The project retains and modifies a small amount of execution infrastructure
originally published as part of FlashInfer-Bench under Apache-2.0. The retained
ideas and code are limited to generic source loading/building, runnable
execution, device abstraction, timing, and isolated evaluation infrastructure.
The original dataset schema, product CLI, agents, Web application, trace
management, and domain-specific evaluators are not included.

The new schema, PyTree protocol, HTTP service, heterogeneous accelerator
adaptations, profiling integrations, and most evaluation behavior were
developed in this project or in its predecessor development branch.

## Bundled third-party source

`kernelgen_server/profiling/hygon/hygon_roctracer_compat/` vendors a small
ROCm 6.1 ROCtracer tool/file-plugin compatibility subset originally published
by Advanced Micro Devices, Inc. It is distributed under the MIT license kept
in that directory. The local changes adapt the loader and activity handling to
Hygon DTK without modifying the system DTK installation.

`data/kernelbench/` contains generated v6.2 contracts and reference code
derived from KernelBench Level 1–3 tasks. KernelBench is distributed under the
MIT license preserved in `licenses/KernelBench-MIT.txt` and copied into the
generated Catalog as `data/kernelbench/LICENSE`.

## External vendor tools

Hardware profiler executables and SDKs such as NCU, msprof, MCU, CNPerf,
hipprof/rocprof, mcTracer, ixsys/IXKN, XProfiler, ACU and topsprof are supplied
by their respective accelerator images or host installations. They are invoked
by adapters in this repository but are not redistributed as part of the Python
package. Their licenses and redistribution terms remain those of their vendors.

Modified derived files must preserve the original copyright notice and state
that they were changed. The top-level `LICENSE` and `NOTICE` must be included
in source and binary distributions.

The archived `data/.old/flaggems-v5` catalog is derived from FlagGems tests
and benchmarks, also distributed under Apache-2.0. The top-level `NOTICE`
records its provenance.
