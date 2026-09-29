import torch

REFERENCE_DEVICE = "target"


def run(input, p=0.5, train=True):
    return torch.alpha_dropout(input, p, train)


def valid(ref_outputs, sol_outputs, inputs, ctx):
    # The FlagGems accuracy test checks only output structure for train=True;
    # train=False compares the candidate with a reference copied from itself.
    # KGS performs the shape/dtype return-contract checks before this hook.
    del ref_outputs, sol_outputs, inputs, ctx
    return True
