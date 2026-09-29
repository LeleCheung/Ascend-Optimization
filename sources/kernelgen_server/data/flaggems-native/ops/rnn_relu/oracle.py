import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    inputs = ctx["inputs"]
    case = inputs.get("case")
    if case is None:
        return None

    device_type = torch.device(device).type
    if device_type == "musa":
        torch.backends.mudnn.allow_tf32 = False
    elif device_type == "npu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        try:
            _tb = getattr(torch.backends, device_type)
            _tb.matmul.allow_tf32 = False
        except Exception:
            pass

    seed = ctx["seed"]
    dtype = getattr(torch, case["dtype"])
    phase = case["phase"]

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    factory_device = generator.device
    generator.manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)

    if phase == "timing":
        shape = case["shape"]
        inp = randn(*shape["input"])
        hx = randn(*shape["hx"])
        params = tuple(randn(*s) for s in shape["params"])
        cp = case["params"]
        has_biases = cp["has_biases"]
        num_layers = cp["num_layers"]
        dropout = cp["dropout"]
        train = cp["train"]
        bidirectional = cp["bidirectional"]
        batch_first = cp["batch_first"]
    else:
        seq_len = case["seq_len"]
        batch_size = case["batch_size"]
        input_size = case["input_size"]
        hidden_size = case["hidden_size"]
        batch_first = case["batch_first"]
        has_biases = True
        num_layers = 1
        dropout = 0.0
        train = False
        bidirectional = False

        if batch_first:
            inp = randn(batch_size, seq_len, input_size)
        else:
            inp = randn(seq_len, batch_size, input_size)
        hx = randn(1, batch_size, hidden_size)
        params = (
            randn(hidden_size, input_size),
            randn(hidden_size, hidden_size),
            randn(hidden_size),
            randn(hidden_size),
        )

    target_device = torch.device(device)
    if factory_device != target_device:
        inp = inp.to(device)
        hx = hx.to(device)
        params = tuple(p.to(device) for p in params)

    return {
        "input": inp,
        "hx": hx,
        "params": params,
        "has_biases": has_biases,
        "num_layers": num_layers,
        "dropout": dropout,
        "train": train,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
    }


def run(input, hx=None, params=None, has_biases=True, num_layers=1,
        dropout=0.0, train=False, bidirectional=False, batch_first=False):
    return torch.rnn_relu(input, hx, params, has_biases, num_layers,
                          dropout, train, bidirectional, batch_first)
