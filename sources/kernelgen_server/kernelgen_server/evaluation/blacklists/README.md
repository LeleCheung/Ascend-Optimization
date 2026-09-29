# Anti-hack blacklists

Machine-generated enumerations of the **forbidden compute callables** for each
Torch namespace.  They back the blacklist-based detector in
[`../hack_detection.py`](../hack_detection.py), which labels submitted
`language="triton"` implementations that secretly compute through Torch
operators instead of a real `@triton.jit` kernel.

## Files

| File | Namespace | `call_prefix` | Forbidden entries |
|------|-----------|---------------|-------------------|
| `torch.yaml` | `torch` | `torch.` | 688 |
| `tensor_methods.yaml` | `torch.Tensor` | *(empty — method form)* | 503 |
| `nn_functional.yaml` | `torch.nn.functional` | `F.` | 138 |
| `linalg.yaml` | `torch.linalg` | `torch.linalg.` | 41 |
| `fft.yaml` | `torch.fft` | `torch.fft.` | 22 |

Counts are for torch 2.12.0+cu130. Regenerate after a Torch upgrade.

## Layout

Each YAML has a fixed, machine-friendly shape:

```yaml
namespace: "torch"
call_prefix: "torch."
forbidden:
  - matmul
  - softmax
  - ...
```

`hack_detection.py` parses these with a dependency-free line reader (no PyYAML
in the detection path). Keep the layout stable: `namespace:` / `call_prefix:`
scalars, then a `forbidden:` list of `  - name` items.

## How the lists are generated

See [`_generate.py`](_generate.py). Strategy:

1. Enumerate every **public callable** in each namespace via `dir()`, keeping
   only real functions/builtins — classes (exceptions, Storage, type-system
   internals) and submodules are filtered out with `inspect.isclass` and
   `types.ModuleType`.
2. **Subtract an allow-set** of legitimate plumbing that must never be flagged:
   allocation (`empty`, `zeros`, `*_like`), dtype/device movement (`to`,
   `cuda`, `float`), shape/layout (`reshape`, `view`, `permute`,
   `contiguous`), metadata (`size`, `stride`, `numel`), and autograd flags.
3. What remains is the "forbidden compute" blacklist.

`nn.functional`, `linalg`, and `fft` are treated as **all-compute**: their
allow-sets are empty, so every public callable is forbidden.

To regenerate:

```bash
/usr/bin/python _generate.py        # writes /tmp/blacklist_*.yaml
# review the diff, then copy the ones you want over the files here
```

## What the detector matches

* **Namespace form** — `torch.matmul(a, b)`, `F.relu(x)`,
  `torch.linalg.svd(x)`. Import aliases are resolved, so `import torch as tr;
  tr.matmul(...)` and `from torch import matmul as mm; mm(...)` both
  canonicalise to `torch.matmul`.
* **Method form** — `x.softmax(-1)`. Matched by bare attribute name against
  `tensor_methods.yaml`. Best-effort: a same-named method on a non-Tensor
  object would also match, which is why the detector only *labels* and never
  blocks.
* **Wholesale prefixes** — `torch.ops.*`, `torch._C.*`, `torch.overrides.*`
  are forbidden entirely (dynamic-dispatch escape hatches, not enumerable).
  These live in `_FORBIDDEN_PREFIXES` in `hack_detection.py`, not in a YAML.

## Known limitations

* **Blacklists drift with Torch versions.** New ops added upstream are not
  caught until the YAMLs are regenerated. This is the inherent cost of the
  blacklist direction (a whitelist would not drift, but was ruled out for this
  deployment).
* **Method form is heuristic.** The receiver type is unknown at parse time.
