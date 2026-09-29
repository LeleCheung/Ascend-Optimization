# Hygon ROCtracer compatibility layer

This directory vendors the ROCm 6.1 ROCtracer tool and file-plugin sources under their
MIT license. DTK 25.04 exposes the compatible ROCtracer 4.1 API from `libgalaxyhip`, but
its `rocprof` package does not ship the `libroctracer64.so.4` and
`libroctracer_tool.so` paths expected by the launcher.

`kernelgen_server.profiling.hygon.hygon_trace_compat` builds these sources against the local
DTK headers and libraries, creates a private `libroctracer64.so.4` alias, and renders a
private rocprof launcher. Nothing under `/opt/dtk` is modified.

Hygon-specific changes are intentionally small:

- use `hip_api_data_t.begin_timestamp` because DTK callbacks leave `phase_data` null;
- locate the already-loaded `libgalaxyhip.so` instead of AMD's `libamdhip64.so`;
- do not dereference DTK activity `kernel_name`; dispatch names are correlated from HIP
  API records, while uncorrelated copy/memset activities are omitted so the stock
  `tblextr.py` remains usable.
