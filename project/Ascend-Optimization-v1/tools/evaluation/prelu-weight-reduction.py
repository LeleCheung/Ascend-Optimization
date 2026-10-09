"""补齐旧 PReLU 权重归约；采用与固定候选一致的 FP32 分阶段求和顺序。"""


@triton.jit
def _prelu_weight_partial_kernel(values, partials, count, n_iters, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * n_iters)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(0, n_iters):
        offsets = base + j * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            value = tl.load(values + offsets, mask=offsets < count, other=0.0)
        else:
            value = tl.load(values + offsets)
        acc += value.to(tl.float32)
    tl.store(partials + pid, tl.sum(acc, axis=0))


@triton.jit
def _prelu_weight_final_kernel(partials, output, count, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, count, BLOCK):
        offsets = start + lanes
        acc += tl.load(partials + offsets, mask=offsets < count, other=0.0)
    tl.store(output, tl.sum(acc, axis=0))


def _prelu_stable_scalar_sum(values, weight):
    # 与现有候选的归约跨度和计算顺序一致，仅从原逐元素临时缓冲读取。
    block, iters = (4096, 4) if values.numel() <= (1 << 21) else (4096, 8)
    span = block * iters
    count = triton.cdiv(values.numel(), span)
    partials = torch.empty((count,), dtype=torch.float32, device=values.device)
    output = torch.empty_like(weight)
    reduce_block = 128
    while reduce_block < 4096 and reduce_block < count:
        reduce_block *= 2
    _prelu_weight_partial_kernel[(count,)](values, partials, values.numel(), iters,
        BLOCK=block, NEED_MASK=(values.numel() % span) != 0)
    _prelu_weight_final_kernel[(1,)](partials, output, count, BLOCK=reduce_block)
    return output
