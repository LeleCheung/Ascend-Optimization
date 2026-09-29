"""Generate anti-hack blacklist YAML files from torch introspection.

Strategy: enumerate every public callable in each torch namespace, then SUBTRACT
a small allow-set of legitimate helpers (allocation, dtype, shape/layout, device,
autograd/no-op utilities). What remains is the "forbidden compute" blacklist.

This keeps the blacklist form the user asked for, while avoiding false positives
on routine tensor plumbing like x.reshape() / x.to() / x.contiguous().
"""

import torch
import torch.nn.functional as F
import torch.linalg
import torch.fft


import inspect
import types


def public_callables(mod):
    """Public callables that are actual functions/ops, not classes or modules.

    Excludes:
      * names starting with "_"
      * submodules (types.ModuleType) -- e.g. torch.nn, torch.cuda
      * classes -- exceptions, type-system internals, Storage/Tensor subclasses
        (heuristic: inspect.isclass); real ops are functions/builtins, not classes.
    """
    out = []
    for n in dir(mod):
        if n.startswith("_"):
            continue
        try:
            attr = getattr(mod, n)
        except Exception:
            continue
        if not callable(attr):
            continue
        if isinstance(attr, types.ModuleType):
            continue
        if inspect.isclass(attr):
            continue
        out.append(n)
    return sorted(out)


# ---- Allow-sets: legitimate helpers that must NOT be blacklisted ----

# torch.* top-level: allocation, dtype constructors, device/cuda setup, rng,
# grad-mode context managers, and misc non-compute utilities.
ALLOW_TORCH_TOP = {
    # allocation
    "empty", "zeros", "ones", "randn", "rand", "randint", "arange", "linspace",
    "logspace", "tensor", "as_tensor", "from_numpy", "empty_like", "zeros_like",
    "ones_like", "randn_like", "rand_like", "randint_like", "full", "full_like",
    "empty_strided", "eye",
    # dtype / type constructors
    "Size", "dtype", "device", "Tensor", "finfo", "iinfo",
    # device / cuda / rng setup
    "manual_seed", "seed", "set_default_device", "set_default_dtype",
    "get_default_dtype", "set_grad_enabled", "is_grad_enabled",
    "no_grad", "enable_grad", "inference_mode", "autocast",
    "set_num_threads", "get_num_threads",
    # introspection (no compute)
    "is_tensor", "numel", "typename", "is_storage", "is_floating_point",
    "is_complex", "result_type", "promote_types", "can_cast",
    "compile", "jit", "use_deterministic_algorithms",
}

# Tensor methods: shape/layout/dtype/device movement and metadata are legitimate.
ALLOW_TENSOR_METHODS = {
    # dtype / device movement
    "to", "cuda", "cpu", "type", "type_as", "float", "double", "half",
    "bfloat16", "int", "long", "short", "bool", "byte", "char",
    # shape / layout (no arithmetic)
    "reshape", "reshape_as", "view", "view_as", "flatten", "unflatten",
    "squeeze", "unsqueeze", "permute", "transpose", "t", "movedim", "moveaxis",
    "expand", "expand_as", "broadcast_to", "contiguous", "clone", "detach",
    "ravel", "flip", "roll", "narrow", "unfold", "chunk", "split", "unbind",
    # metadata / introspection
    "size", "dim", "numel", "element_size", "stride", "storage_offset",
    "is_contiguous", "is_cuda", "is_floating_point", "is_complex", "is_signed",
    "get_device", "nelement", "ndimension", "item", "tolist",
    # allocation-ish helpers on a tensor
    "new_empty", "new_zeros", "new_ones", "new_full", "new_tensor",
    "empty_like", "zeros_like", "ones_like",
    # autograd plumbing
    "requires_grad_", "retain_grad", "register_hook",
    # copy (movement, not compute)
    "copy_", "fill_", "zero_",
}

ALLOW_FUNCTIONAL = set()   # nn.functional is essentially all compute -> forbid all
ALLOW_LINALG = set()       # linalg is all compute -> forbid all
ALLOW_FFT = set()          # fft is all compute -> forbid all


def emit(path, namespace, prefix, names, allow):
    forbidden = [n for n in names if n not in allow]
    with open(path, "w") as f:
        f.write(f"# Auto-generated from torch {torch.__version__} introspection.\n")
        f.write(f"# Namespace: {namespace}\n")
        f.write(f"# Policy: blacklist. Every entry below is a FORBIDDEN compute call.\n")
        f.write(f"# Legitimate helpers (allocation/dtype/shape/device) were excluded.\n")
        f.write(f"# Total public callables: {len(names)}  |  allowed-excluded: {len(names) - len(forbidden)}  |  forbidden: {len(forbidden)}\n")
        f.write(f"namespace: \"{namespace}\"\n")
        f.write(f"call_prefix: \"{prefix}\"\n")
        f.write(f"forbidden:\n")
        for n in forbidden:
            f.write(f"  - {n}\n")
    print(f"{path}: {len(forbidden)} forbidden (of {len(names)} callables)")


emit("/tmp/blacklist_torch.yaml", "torch", "torch.",
     public_callables(torch), ALLOW_TORCH_TOP)
emit("/tmp/blacklist_tensor_methods.yaml", "torch.Tensor", "",
     public_callables(torch.Tensor), ALLOW_TENSOR_METHODS)
emit("/tmp/blacklist_nn_functional.yaml", "torch.nn.functional", "F.",
     public_callables(F), ALLOW_FUNCTIONAL)
emit("/tmp/blacklist_linalg.yaml", "torch.linalg", "torch.linalg.",
     public_callables(torch.linalg), ALLOW_LINALG)
emit("/tmp/blacklist_fft.yaml", "torch.fft", "torch.fft.",
     public_callables(torch.fft), ALLOW_FFT)
