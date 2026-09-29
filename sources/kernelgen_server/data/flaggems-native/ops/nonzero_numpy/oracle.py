import torch

REFERENCE_DEVICE = "target"
VALID_OWNS_RETURN_CONTRACT = True


def run(inp):
    return torch.ops.aten.nonzero_numpy(inp)


def valid(ref_outputs, sol_outputs, inputs, ctx):
    del inputs, ctx
    if len(sol_outputs) != len(ref_outputs):
        return {
            "passed": False,
            "message": "Number of output tensors should match",
        }
    for index, (sol, ref) in enumerate(zip(sol_outputs, ref_outputs)):
        if not isinstance(sol, torch.Tensor) or not isinstance(ref, torch.Tensor):
            return {
                "passed": False,
                "message": f"output {index} is not a Tensor",
            }
        if sol.shape != ref.shape:
            return {
                "passed": False,
                "message": f"output {index} shape differs",
            }
        if sol.dtype != ref.dtype:
            return {
                "passed": False,
                "message": f"output {index} dtype differs",
            }
        if not torch.equal(sol, ref):
            return {
                "passed": False,
                "message": f"output {index} values differ",
            }
    return True


def gen_inputs(ctx, device):
    inp_spec = ctx["inputs"].get("inp")
    # Ordinary recipe dict — materialiser handles it directly.
    if isinstance(inp_spec, dict) and "type" in inp_spec:
        return None

    case = inp_spec
    phase = case["phase"]
    dtype_str = case["dtype"]
    shape = tuple(case["shape"])
    dtype = getattr(torch, dtype_str)

    if phase == "correctness":
        # Source: torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        try:
            generator = torch.Generator(device=device)
        except Exception:
            generator = torch.Generator(device="cpu")
        generator.manual_seed(ctx["seed"])
        factory_device = generator.device
        inp = torch.randint(-3, 3, shape, device=factory_device, generator=generator)
        inp = inp.to(device).to(dtype)
        return {"inp": inp}

    # timing phase: source allocates int/bool on CPU then moves to device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ctx["seed"])
    factory_device = generator.device

    if dtype == torch.bool:
        # Source: torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=factory_device,
                            generator=generator).to(device)
    else:
        # int16/int32: torch.randint(iinfo.min, iinfo.max, shape, dtype=dtype, device="cpu").to(device)
        info = torch.iinfo(dtype)
        inp = torch.randint(info.min, info.max, shape, dtype=dtype, device=factory_device,
                            generator=generator).to(device)

    return {"inp": inp}
