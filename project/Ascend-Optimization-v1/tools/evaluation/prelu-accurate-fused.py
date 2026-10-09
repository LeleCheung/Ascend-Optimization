"""FP32 PReLU 输入梯度与误差补偿权重梯度部分和融合计算。"""


@triton.jit
def _prelu_accurate_fused_kernel(g_ptr, x_ptr, w_ptr, gi_ptr, highs, lows,
                                 count, n_iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    weight = tl.load(w_ptr)
    high = tl.full((), 0.0, tl.float32)
    low = tl.full((), 0.0, tl.float32)
    for j in range(0, n_iters):
        offsets = (pid * n_iters + j) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(gi_ptr + offsets, tl.where(x > 0, g, g * weight), mask=mask)
        products = tl.where(x < 0, g * x, 0.0)
        block_high, block_low = _prelu_pair_sum(products, tl.zeros([BLOCK], tl.float32), BLOCK)
        new_high, error = _prelu_two_sum(high, block_high)
        high, low = _prelu_two_sum(new_high, low + block_low + error)
    tl.store(highs + pid, high)
    tl.store(lows + pid, low)


_prelu_previous_run = run


def run(grad_output, self, weight):
    if self.dtype != torch.float32 or weight.numel() != 1 or self.numel() == 0:
        return _prelu_previous_run(grad_output, self, weight)
    x = self.contiguous()
    g = grad_output.contiguous()
    gi = torch.empty_like(x)
    gw = torch.empty_like(weight)
    block = 1024
    iters = 16 if x.numel() <= (1 << 21) else 32
    count = triton.cdiv(x.numel(), block * iters)
    highs = torch.empty((count,), dtype=torch.float32, device=x.device)
    lows = torch.empty_like(highs)
    _prelu_accurate_fused_kernel[(count,)](g, x, weight, gi, highs, lows, x.numel(), iters,
                                         BLOCK=block, enable_fp_fusion=False)
    _prelu_final_sum_kernel[(1,)](highs, lows, gw, count, BLOCK=block, enable_fp_fusion=False)
    return gi, gw
