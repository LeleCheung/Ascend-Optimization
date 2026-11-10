import torch
import triton
import triton.language as tl


@triton.jit
def _probe_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(x_ptr + offs, v + 1.0, mask=m)


def _probe_host(x):
    a = torch.empty_like(x)
    b = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    c = torch.zeros(4, device=x.device, dtype=torch.int32)
    d = torch.ones(4, device=x.device, dtype=torch.int32)
    e = x.cpu()
    f = x.item()
    g = x.tolist()
    h = x.numpy()
    i = x.to("cpu")
    j = torch.equal(a, b)
    k = torch.allclose(a, b)
    l = x.view(-1)
    m = x.reshape(-1)
    n = x.transpose(0, 1)
    o = torch.as_strided(x, x.shape, x.stride())
    p = x.zero_()
    q = x.to(torch.float32)
    r = triton.cdiv(8, 4)
    s = x.data_ptr()
    t = x.stride()
    u = x.is_contiguous()
    v = x.numel()
    w = x.device
    y = triton.runtime.driver.active.get_current_stream(0)
    z = x.fill_(1.0)
    return (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, y, z)


def run(A):
    n = A.numel()
    _probe_kernel[(triton.cdiv(n, 1024),)](A, n, BLOCK=1024)
    return A
