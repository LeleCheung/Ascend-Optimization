"""Generate anti-hack graylist YAML files from torch introspection.

A *graylist* is the complement of the matching ``blacklists/*.yaml`` file: for
each namespace it enumerates the public callables that torch actually exposes
but that the blacklist deliberately does NOT forbid.  These are the legitimate
plumbing helpers (allocation, dtype/device movement, shape/layout, metadata,
autograd flags) that the blacklist's allow-set carved out.

    graylist(namespace) = public_callables(namespace) - forbidden(blacklist)

They are a *weak* signal, not a fallback: a Triton implementation is free to
call ``x.reshape(...)`` or ``torch.empty(...)``, but surfacing which plumbing a
submission leans on is useful review context, so the detector labels graylist
hits separately from (and far more softly than) blacklist hits.

To regenerate:

    /usr/bin/python _generate.py        # writes /tmp/graylist_*.yaml
    # review the diff, then copy the ones you want over the files here
"""

import inspect
import types
from pathlib import Path

import torch
import torch.fft
import torch.linalg
import torch.nn.functional as F

_BLACKLIST_DIR = Path(__file__).resolve().parent.parent / "blacklists"


def public_callables(mod):
    """Public callables that are real functions/ops, not classes or modules.

    Mirrors ``blacklists/_generate.py`` exactly so the graylist is a clean
    complement of the blacklist over the same candidate set.
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


def read_forbidden(path):
    """Read the ``forbidden:`` list out of a blacklist YAML."""
    forbidden = set()
    in_forbidden = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("forbidden:"):
            in_forbidden = True
        elif line[:1] not in (" ", "-") and ":" in line:
            in_forbidden = False
        elif in_forbidden and line.lstrip().startswith("- "):
            forbidden.add(line.lstrip()[2:].strip())
    return forbidden


# Operations that should NOT be in graylist:
#   1. Tensor creation ops -- clear allocation calls, not "ambiguous view/layout plumbing"
#   2. dtype/device conversion ops -- .to() returns a COPY when dtype/device differs,
#      making it a conversion/compute op, not a pure view.  A Triton impl that calls
#      x.to(torch.float32) is doing an element-wise cast, not layout manipulation.
_GRAY_EXCLUDED = frozenset([
    # Allocation (zeros/ones/empty family)
    "empty", "empty_like", "empty_permuted", "empty_quantized", "empty_strided",
    "zeros", "ones", "full", "zeros_like", "ones_like", "full_like",
    "eye", "arange", "range", "linspace", "logspace",
    # Random
    "rand", "rand_like", "randn", "randn_like", "randint", "randint_like",
    "manual_seed", "seed",
    # Tensor construction
    "tensor", "as_tensor", "from_numpy", "asarray", "scalar_tensor",
    # Sparse constructors
    "sparse_coo_tensor", "sparse_csr_tensor", "sparse_csc_tensor",
    "sparse_bsr_tensor", "sparse_bsc_tensor", "sparse_compressed_tensor",
    # Window functions (signal processing, not layout)
    "bartlett_window", "blackman_window", "hamming_window", "hann_window", "kaiser_window",
    # dtype/device conversion -- returns a *copy* when conversion is needed;
    # this is a data transformation, not a view or layout operation.
    "to",
])


def emit(out_path, namespace, prefix, names, forbidden):
    gray = [n for n in names if n not in forbidden and n not in _GRAY_EXCLUDED]
    with open(out_path, "w") as f:
        f.write(f"# Auto-generated from torch {torch.__version__} introspection.\n")
        f.write(f"# Namespace: {namespace}\n")
        f.write(f"# Policy: graylist for ambiguous view/layout/metadata ops only.\n")
        f.write(f"# Tensor creation (zeros/ones/empty/rand) is excluded -- those\n")
        f.write(f"# belong in the blacklist, not the graylist.\n")
        f.write(f"# graylist = public_callables - blacklist.forbidden - GRAY_EXCLUDED.\n")
        f.write(f"# Total public callables: {len(names)}  |  blacklisted: {len(names) - len(gray) - len([n for n in names if n in _GRAY_EXCLUDED])}  |  gray: {len(gray)}\n")
        f.write(f'namespace: "{namespace}"\n')
        f.write(f'call_prefix: "{prefix}"\n')
        f.write("gray:\n")
        for n in gray:
            f.write(f"  - {n}\n")
    excluded_count = len([n for n in names if n in _GRAY_EXCLUDED and n not in forbidden])
    print(f"{out_path}: {len(gray)} gray (excluded {excluded_count} tensor-creation ops from {len(names)} callables)")


SPECS = [
    ("torch.yaml", "graylist_torch.yaml", torch, "torch", "torch."),
    ("tensor_methods.yaml", "graylist_tensor_methods.yaml", torch.Tensor, "torch.Tensor", ""),
    ("nn_functional.yaml", "graylist_nn_functional.yaml", F, "torch.nn.functional", "F."),
    ("linalg.yaml", "graylist_linalg.yaml", torch.linalg, "torch.linalg", "torch.linalg."),
    ("fft.yaml", "graylist_fft.yaml", torch.fft, "torch.fft", "torch.fft."),
]


for bl_name, out_name, mod, namespace, prefix in SPECS:
    forbidden = read_forbidden(_BLACKLIST_DIR / bl_name)
    emit(f"/tmp/{out_name}", namespace, prefix, public_callables(mod), forbidden)
