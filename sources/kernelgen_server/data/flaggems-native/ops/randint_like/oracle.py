import torch

REFERENCE_DEVICE = "target"


def run(self, high, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return torch.randint_like(
        self,
        high,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )


def valid(ref_outputs, sol_outputs, inputs, ctx):
    del ref_outputs, ctx
    high = inputs["high"]
    sol = sol_outputs[0]
    assert (sol >= 0).all(), "Output contains values less than 0"
    assert (sol < high).all(), f"Output contains values not less than {high}"
    return True
