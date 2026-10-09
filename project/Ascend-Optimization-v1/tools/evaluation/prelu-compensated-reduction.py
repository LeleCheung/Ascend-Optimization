"""为旧 PReLU 补齐稳定的 FP32 权重梯度归约；保留原逐元素输出。"""


@triton.jit
def _prelu_two_sum(a, b):
    total = a + b
    b_virtual = total - a
    error = (a - (total - b_virtual)) + (b - b_virtual)
    return total, error


@triton.jit
def _prelu_pair_sum(high, low, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    for stage in tl.static_range(0, 10):
        peer = lanes ^ (1 << stage)
        other_high = tl.gather(high, peer, axis=0)
        other_low = tl.gather(low, peer, axis=0)
        total, error = _prelu_two_sum(high, other_high)
        low = low + other_low + error
        high, residual = _prelu_two_sum(total, low)
        low = residual
    return tl.sum(tl.where(lanes == 0, high, 0.0)), tl.sum(tl.where(lanes == 0, low, 0.0))


@triton.jit
def _prelu_partial_sum_kernel(values, highs, lows, count, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + offsets, mask=offsets < count, other=0.0).to(tl.float32)
    high, low = _prelu_pair_sum(value, tl.full((BLOCK,), 0.0, tl.float32), BLOCK)
    tl.store(highs + pid, high)
    tl.store(lows + pid, low)


@triton.jit
def _prelu_final_sum_kernel(highs, lows, output, count, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    high = tl.full((BLOCK,), 0.0, tl.float32)
    low = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(0, count, BLOCK):
        offsets = start + lanes
        a = tl.load(highs + offsets, mask=offsets < count, other=0.0)
        b = tl.load(lows + offsets, mask=offsets < count, other=0.0)
        high_new, error = _prelu_two_sum(high, a)
        high, low = _prelu_two_sum(high_new, low + b + error)
    result_high, result_low = _prelu_pair_sum(high, low, BLOCK)
    tl.store(output, result_high + result_low)


def _prelu_stable_scalar_sum(values, weight):
    block = 1024
    count = triton.cdiv(values.numel(), block)
    highs = torch.empty((count,), dtype=torch.float32, device=values.device)
    lows = torch.empty_like(highs)
    output = torch.empty_like(weight)
    _prelu_partial_sum_kernel[(count,)](values, highs, lows, values.numel(), BLOCK=block, enable_fp_fusion=False)
    _prelu_final_sum_kernel[(1,)](highs, lows, output, count, BLOCK=block, enable_fp_fusion=False)
    return output
