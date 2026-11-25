# Operator-specific support

Generic analysis/runtime lives in `hgkda/src/hgkda/`; internal method/regression tools
live in sibling directories. This directory holds targeted checks and native reference
adapters, not separate per-operator optimization controllers.

| Directory | Purpose |
| --- | --- |
| `selu_/` | Offset views, current streams, pointers, and tails |
| `log10_/` | Tiny/special values, numerical boundaries, and tails |
| `negative_/` | Actual cached Triton artifacts and dynamic inputs |
| `uniform/` | Native composed Uniform+Clamp reference |
| `expm1_/` | Official precision modes and small-value qualification |

Checks are separate from full KGS acceptance. Run NPU actions only in the matching
Linux/CANN environment on allocated project devices. Public CLI actions are listed
by `python -m hgkda.runtime --help`; internal validation actions by
`python -m development.tools.skill_validation --help`.

Old direct candidate builders are in [archive](../archive/README.md). CC/DeepSeek owns
future candidate and launch-wrapper changes.
