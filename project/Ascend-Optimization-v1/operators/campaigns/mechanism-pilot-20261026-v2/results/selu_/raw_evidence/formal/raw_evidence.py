import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


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


def _pick_block(n_elements):
    # 同一JIT分派路径下的分块干预表明：1024元素时单程序块最省；
    # 大输入改为较大BLOCK降低program数量与调度/掩码开销，上限4096避免寄存器压力。
    if n_elements <= 4096:
        block = 1024
    elif n_elements <= 262144:
        block = 4096
    else:
        block = 4096
    # 保持grid不超过65535
    while block < 65536 and triton.cdiv(n_elements, block) > 65535:
        block *= 2
    return block


def selu_(*args, **kwargs):
    logger.debug("GEMS SELU_")
    x = None
    if len(args) > 0 and torch.is_tensor(args[0]):
        x = args[0]
    elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
        x = kwargs["input"]
    elif "self" in kwargs and torch.is_tensor(kwargs["self"]):
        x = kwargs["self"]
    elif "x" in kwargs and torch.is_tensor(kwargs["x"]):
        x = kwargs["x"]
    else:
        raise ValueError(
            "selu_ expects a Tensor as the first argument or under 'input'/'self'/'x' keyword."
        )

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not x.is_contiguous()) or (x.dtype not in supported_dtypes):
        torch.ops.aten.selu_(x)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = _pick_block(n_elements)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        selu_kernel_[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x

_baseline = selu_


def run(A):
    return _baseline(A)


selu_ = run
