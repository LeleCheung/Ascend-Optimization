import logging

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ALPHA = 1.6732632423543772
_SCALE = 1.0507009873554805


@triton.jit
def selu_kernel_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
    y = y_f32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def _direct_call(compiled, grid, stream, bound_args):
    gx, gy, gz = grid
    metadata = compiled.launch_metadata(grid, stream, *bound_args.values())
    compiled.run(
        gx, gy, gz, stream, compiled.function, compiled.packed_metadata, metadata,
        triton.knobs.runtime.launch_enter_hook, triton.knobs.runtime.launch_exit_hook,
        *bound_args.values(),
    )


def _launch(x, n_elements, block, grid):
    dev = x.device
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)

    # Normal JIT path: produces (or reuses) the specialization for this call.
    compiled = selu_kernel_[grid](x, n_elements, BLOCK_SIZE=block)
    if hasattr(compiled, "result"):
        compiled = compiled.result()

    cache = selu_kernel_.device_caches.get(device)
    if cache is not None and len(cache) == 5:
        _, _, _, _, binder = cache
        kwargs = {"BLOCK_SIZE": block}
        kwargs["debug"] = kwargs.get("debug", selu_kernel_.debug) or triton.knobs.runtime.debug
        _flagprism = getattr(triton.runtime.jit, "_flagprism", None)
        if _flagprism is not None:
            _flagprism.apply_compile_options(kwargs)
        bound_args, _spec, _opts = binder(x, n_elements, **kwargs)
        # Fast path: same compiled artifact and same bound pointer of this call.
        if tuple(bound_args.values()) == tuple([x, n_elements, block]):
            _direct_call(compiled, grid, stream, bound_args)
            return

    selu_kernel_[grid](x, n_elements, BLOCK_SIZE=block)


def selu_(x):
    if not torch.is_tensor(x):
        raise ValueError("selu_ expects a Tensor as the first argument.")

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not x.is_contiguous()) or (x.dtype not in supported_dtypes):
        out = torch.ops.aten.selu_(x)
        return x if out is None else x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    if n_elements <= 1024:
        block = 1024
    elif n_elements <= 8192:
        block = 2048
    else:
        block = 4096
    grid = (triton.cdiv(n_elements, block), 1, 1)

    device = x.device
    with torch.npu.device(device):
        _launch(x, n_elements, block, grid)
    return x


run = selu_
