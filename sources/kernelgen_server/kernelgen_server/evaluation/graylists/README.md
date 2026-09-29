# Anti-hack graylists

Machine-generated enumerations of **ambiguous view/layout operations** for each
Torch namespace — the complement of the matching
[`../blacklists/*.yaml`](../blacklists/) file, with explicit exclusion of tensor
creation operations:

```
graylist(namespace) = public_callables(namespace) - blacklist.forbidden(namespace) - tensor_creation_ops
```

**IMPORTANT**: Tensor creation operations (zeros, ones, empty, rand, arange,
linspace, from_numpy, etc.) are **explicitly excluded** from the graylist. These
are not "ambiguous plumbing" but clear allocation calls that a Triton
implementation should handle via `tl.zeros` / pre-allocated outputs, not by
calling torch APIs.

The graylist focuses on truly ambiguous operations: reshape, transpose, stride,
copy, broadcast, contiguous, view, metadata queries (`x.numel()`, `.dtype`,
`.device`). A `language="triton"` submission is free to use these, so a graylist
hit is a **weak, informational signal**, not a fallback. The detector in
[`../graylist_detection.py`](../graylist_detection.py) surfaces graylist matches
separately from blacklist matches and — unlike the blacklist — **never sets
`is_hack`** on their account.

## Files

| File | Namespace | `call_prefix` | Gray entries | Notes |
|------|-----------|---------------|--------------|-------|
| `torch.yaml` | `torch` | `torch.` | 151 | Excludes 40 tensor-creation ops |
| `tensor_methods.yaml` | `torch.Tensor` | *(empty — method form)* | 190 | |
| `nn_functional.yaml` | `torch.nn.functional` | `F.` | 3 | |
| `linalg.yaml` | `torch.linalg` | `torch.linalg.` | 1 | |
| `fft.yaml` | `torch.fft` | `torch.fft.` | 0 | |

Counts are for torch 2.12.0+cu130. Regenerate after a Torch upgrade.

## Layout

Same fixed shape as the blacklists, except the list key is `gray:` (not
`forbidden:`). `hack_detection.py` parses these with the same dependency-free
line reader (no PyYAML in the detection path). Keep the layout stable:
`namespace:` / `call_prefix:` scalars, then a `gray:` list of `  - name` items.

```yaml
namespace: "torch"
call_prefix: "torch."
gray:
  - clone
  - reshape
  - transpose
  - ...
```

**Note**: Tensor creation operations (zeros, ones, empty, rand, arange, etc.)
are intentionally absent — see `_GRAY_EXCLUDED` in [`_generate.py`](_generate.py).

## How the lists are generated

See [`_generate.py`](_generate.py). It reuses the exact `public_callables()`
candidate enumeration from `../blacklists/_generate.py`, reads the `forbidden:`
list out of each blacklist YAML, and emits the set difference.

To regenerate:

```bash
/usr/bin/python _generate.py        # writes /tmp/graylist_*.yaml
# review the diff, then copy the ones you want over the files here
```

## What the detector matches

Same namespace / method / alias resolution as the blacklist path. The only
difference is severity: graylist hits are reported under a separate
`graylist Torch API` / `graylist Tensor method` label and do **not** flip
`is_hack`.
