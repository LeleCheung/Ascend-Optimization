import math
import torch

REFERENCE_DEVICE = "target"


def run(self, mean=1.0, std=2.0, *, generator=None):
    """In-place log-normal fill; mutates self and returns it."""
    self.log_normal_(mean, std, generator=generator)
    return self


def valid(ref_outputs, sol_outputs, inputs, ctx):
    """Statistical validation for the stochastic log_normal_ operation.

    Mirrors the pytest assertions:
      1. All output values must be positive.
      2. The sample mean must lie within 15 % of the theoretical mean
         exp(mean + std**2 / 2), checked at float32 precision with
         rtol=1.3e-6 (RESOLUTION[float32]).
    """
    mean_param = inputs.get("mean", 1.0)
    std_param = inputs.get("std", 2.0)

    sol = sol_outputs[0].cpu()

    # All log-normal samples are strictly positive
    assert (sol > 0).all(), "log_normal_ produced non-positive values"

    # Statistical mean check
    mean_res = torch.mean(sol.to(torch.float32))
    expected_mean = math.exp(mean_param + std_param ** 2 / 2)
    mean_tol = 0.15 * expected_mean
    rtol = 1.3e-6  # RESOLUTION[torch.float32]
    abs_diff = abs(mean_res.item() - expected_mean)
    allowed = mean_tol + rtol * abs(expected_mean)
    assert abs_diff <= allowed, (
        f"sample mean {mean_res.item():.6f} deviates from expected "
        f"{expected_mean:.6f} by {abs_diff:.6f}, tolerance {allowed:.6f}"
    )
    return True
