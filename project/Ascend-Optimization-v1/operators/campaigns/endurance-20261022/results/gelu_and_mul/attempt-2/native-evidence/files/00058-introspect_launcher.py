"""Read-only introspection of the installed FlagTree/Triton launch ABI."""

import inspect
import json
import os

import triton
from triton.compiler.compiler import CompiledKernel
from triton.backends.ascend import driver as asc_driver

OUT = {}


def snippet(obj, start, end):
    try:
        src, first = inspect.getsourcelines(obj)
        return "".join(src[max(0, start - first): end - first])
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"[:200]


OUT["npu_driver_call_source"] = snippet(asc_driver.NPUDriver.__call__, 0, 60)
try:
    OUT["npu_driver_call_sig"] = str(inspect.signature(asc_driver.NPUDriver.__call__))
except Exception as exc:
    OUT["npu_driver_call_sig"] = str(exc)

OUT["compiled_kernel_getitem"] = snippet(CompiledKernel.__getitem__, 0, 40)
try:
    OUT["compiled_kernel_run_sig"] = str(inspect.signature(CompiledKernel.run))
except Exception as exc:
    OUT["compiled_kernel_run_sig"] = str(exc)

try:
    import triton.spec.ascend.compiler.compiler as asc_comp
    OUT["ascend_compiler_module"] = asc_comp.__file__
except Exception as exc:
    OUT["ascend_compiler_module"] = f"{type(exc).__name__}: {exc}"

try:
    from triton.runtime.jit import JITFunction

    src, first = inspect.getsourcelines(JITFunction.run)
    OUT["jit_run_source_len"] = len(src)
    OUT["jit_run_source"] = "".join(src)
except Exception as exc:
    OUT["jit_run_source"] = f"{type(exc).__name__}: {exc}"[:300]

out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "launcher_abi.json"), "w") as fh:
    json.dump(OUT, fh, indent=2)
print(json.dumps({k: (v if len(str(v)) < 3000 else str(v)[:3000]) for k, v in OUT.items()}, indent=2))
