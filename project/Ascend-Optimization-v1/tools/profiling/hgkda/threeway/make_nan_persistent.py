"""大张量采用固定40个program分段流式处理；小输入及特殊布局保留已验证实现。"""
from make_proposals import BATCH, proposal


def main():
    source = (BATCH/'proposals/nan_to_num_/compact-default-dispatch-v1/main.py').read_text(encoding='utf8')
    source += '''

@triton.jit
def _nan_streaming(A, n, nan_v, pos_v, neg_v,
                   LOWP: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for start in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offset = start + lane
        mask = offset < n
        x = tl.load(A + offset, mask=mask, other=0.0)
        if LOWP:
            dt = A.dtype.element_ty
            y = tl.maximum(tl.minimum(x, pos_v.to(dt)), neg_v.to(dt))
            y = tl.where(x != x, nan_v.to(dt), y)
        else:
            y = tl.where(x != x, nan_v, x)
            y = tl.where(x == float('inf'), pos_v, y)
            y = tl.where(x == float('-inf'), neg_v, y)
        tl.store(A + offset, y, mask=mask)


_compact_run = run
_PERSISTENT = {}


def run(A, nan=None, posinf=None, neginf=None):
    n = A.numel()
    if (n < 1048576 or posinf is not None or neginf is not None
            or A.dtype not in _DEFAULT_LIMITS or not A.is_contiguous()):
        return _compact_run(A, nan, posinf, neginf)
    ptr = A.data_ptr()
    if ptr & 15:
        return _compact_run(A, nan, posinf, neginf)
    dtype = A.dtype
    key = (dtype, n, A.get_device())
    plan = _PERSISTENT.get(key)
    if plan is False:
        return _compact_run(A, nan, posinf, neginf)
    lowp = dtype != torch.float32
    block = 4096
    if plan is None:
        try:
            ck = _nan_streaming.warmup(A,n,0.0,0.0,0.0,LOWP=lowp,
                BLOCK=block,grid=(40,),multibuffer=True)
            launcher = ck.run
            if (getattr(launcher,'compile_only',False) or getattr(launcher,'enable_msprof_register_tensor',False)
                    or getattr(launcher.metadata,'debug_enabled',False) or getattr(launcher.metadata,'debug',False)):
                raise RuntimeError('use registered profiling launcher')
            plan = (launcher.launch,ck.function,ck.packed_metadata,_resolve_stream(A.get_device()))
            _PERSISTENT[key] = plan
        except Exception:
            _PERSISTENT[key] = False
            return _compact_run(A,nan,posinf,neginf)
    limit = _DEFAULT_LIMITS[dtype]
    plan[0](40,1,1,plan[3](),plan[1],plan[2],None,None,None,
            ptr,n,0.0 if nan is None else nan,limit,-limit,int(lowp),block)
    return A
'''
    proposal('nan_to_num_', 'persistent-streaming-v2', source,
             '大连续张量改为40个program循环流式处理4096元素块，启用后端multibuffer；小张量和特殊布局保留原路径',
             '减少逻辑program调度与大UB tile压力，改善最大输入搬运与计算重叠；性能和原地特殊值语义由完整KGS及补测决定')


if __name__ == '__main__':
    main()
