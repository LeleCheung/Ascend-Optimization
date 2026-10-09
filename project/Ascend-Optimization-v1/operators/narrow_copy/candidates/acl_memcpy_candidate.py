"""实验性 CANN D2D copy；仅连续 dim=0，正式 ABI 仍为 run。"""
import ctypes
import torch
import torch_npu

_acl = ctypes.CDLL("libacl_rt.so")
_acl.aclrtMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_acl.aclrtMemcpyAsync.restype = ctypes.c_int
_D2D = 3

def run(inp, dim, start, length):
    if dim < 0: dim += inp.ndim
    if dim != 0 or not inp.is_contiguous():
        return torch.narrow_copy(inp, dim, start, length)
    out_shape = (length, *inp.shape[1:])
    out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    elems = out.numel()
    itemsize = inp.element_size()
    src = inp.data_ptr() + start * inp.stride(0) * itemsize
    stream = torch_npu.npu.current_stream(inp.device).npu_stream
    rc = _acl.aclrtMemcpyAsync(ctypes.c_void_p(out.data_ptr()), elems * itemsize,
                                ctypes.c_void_p(src), elems * itemsize, _D2D,
                                ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"aclrtMemcpyAsync failed: {rc}")
    return out
