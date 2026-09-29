#!/usr/bin/env python
"""Oracle reference for the FlagGems ``unbind_copy`` operator.

The marked accuracy pytest builds its reference with
``torch.unbind_copy(ref_inp, dim)`` (``to_reference`` with ``upcast=False``
is the identity here since ``TO_CPU=False``) and compares the candidate
output against it by asserting equal sequence length and then zipping the
leaves (``flag_gems.testing.assert_equal`` -> ``torch.testing.assert_close``
with ``atol=0, rtol=0, equal_nan=False``).  The Torch reference therefore
keeps its native tuple container, and ``VALID_OWNS_RETURN_CONTRACT``
reproduces the entire source contract (length, Tensor leaves, shape, dtype,
exact values) so the list/tuple container difference of the FlagGems
implementation is intentionally equivalent.  The benchmark times the same
native primitive (``torch.unbind_copy`` with ``dim=0``), so a single shared
``run`` serves both phases.
"""

import torch

REFERENCE_DEVICE = "target"
VALID_OWNS_RETURN_CONTRACT = True


def run(input, dim=0):
    return torch.unbind_copy(input, dim)


def valid(ref_outputs, sol_outputs, inputs, ctx):
    ref_out = ref_outputs[0]
    res_out = sol_outputs[0]

    # assert len(res_out) == len(ref_out)
    if len(res_out) != len(ref_out):
        return False

    # for res, ref in zip(res_out, ref_out):
    #     utils.gems_assert_equal(res, ref)
    # flag_gems.testing.assert_equal -> torch.testing.assert_close(
    #     res, ref, atol=0, rtol=0, equal_nan=False)
    for res, ref in zip(res_out, ref_out):
        if not isinstance(res, torch.Tensor) or not isinstance(ref, torch.Tensor):
            return False
        if res.dtype != ref.dtype:
            return False
        if res.shape != ref.shape:
            return False
        if not torch.equal(res, ref):
            return False
    return True
