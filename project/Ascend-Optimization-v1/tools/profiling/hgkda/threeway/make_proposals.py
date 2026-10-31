"""从冻结起点生成机制明确的候选；只有正式评测及补测通过才可保留。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/threeway-20261030'


def proposal(op,name,body,summary,prediction):
    folder=BATCH/'proposals'/op/name
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/'main.py'
    raw=body.encode('utf8')
    if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
    source.write_bytes(raw)
    value=dict(operator=op,id=name,source='main.py',source_sha256=hashlib.sha256(raw).hexdigest(),
               summary=summary,prediction=prediction,attribution='Codex根据历史KGS逐case结果及宿主诊断提出；本批原KGS与独立补测验收')
    (folder/'proposal.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    get=lambda op:(BATCH/'starting-points'/(op+'.py')).read_text(encoding='utf8')
    cross=get('cross_out')
    start=cross.index('def _pick_block(');end=cross.index('\ndef _default_dim',start)
    cross=cross[:start]+'''def _pick_block(n_elem, esize):
    # Larger low-precision UB tiles reduce programs and repeated index setup.
    if n_elem <= 131072:
        return 1024
    return 1024 if esize == 4 else 4096

'''+cross[end:]
    proposal('cross_out','coarser-lowp-tiles-v1',cross,
             '扩大低精度连续转叉积分块，保留原gather数值顺序、真实stride及out别名保护',
             '低精度大输入program数量约减半；完整调用耗时降低，FP32与布局语义不变')

    nan=get('nan_to_num_').replace('def run(A, nan=None, posinf=None, neginf=None):','def _legacy_run(A, nan=None, posinf=None, neginf=None):',1)
    nan+='''

_DEFAULT_PLANS = {}
_DEFAULT_LIMITS = {torch.float16: 65504.0, torch.bfloat16: 3.3895313892515355e38,
                   torch.float32: 3.4028234663852886e38}


def run(A, nan=None, posinf=None, neginf=None):
    if posinf is not None or neginf is not None:
        return _legacy_run(A, nan, posinf, neginf)
    dtype = A.dtype
    limit = _DEFAULT_LIMITS.get(dtype)
    if limit is None:
        return _legacy_run(A, nan, posinf, neginf)
    n = A.numel()
    if not n:
        return A
    pointer = A.data_ptr()
    if not A.is_contiguous() or pointer & 15:
        return _legacy_run(A, nan, posinf, neginf)
    key = (dtype, n, A.get_device())
    entry = _DEFAULT_PLANS.get(key)
    if entry is None:
        entry = _build_entry(A, n, dtype != torch.float32)
        _DEFAULT_PLANS[key] = entry
    if entry is _FAIL:
        return _legacy_run(A, nan, posinf, neginf)
    entry[0](entry[3], 1, 1, entry[5](), entry[1], entry[2], None, None, None,
             pointer, n, 0.0 if nan is None else nan, limit, -limit, entry[4])
    return A
'''
    proposal('nan_to_num_','compact-default-dispatch-v1',nan,
             '默认上下界直接缓存编译计划，消除重复finfo查询与两层分派；值、地址和当前stream每次读取',
             '小中输入宿主完整调用减少约1到3微秒；自定义边界、非连续和非对齐输入保留旧有效路径')

    norm=get('native_group_norm').replace('def run(input, weight, bias, N, C, HxW, group, eps):','def _legacy_run(input, weight, bias, N, C, HxW, group, eps):',1)
    norm=norm.replace('native_group_norm = run', 'native_group_norm = _legacy_run', 1)
    norm+='''

_HOST_PLANS = {}


def run(input, weight, bias, N, C, HxW, group, eps):
    x = input
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    if not x.is_contiguous():
        return _legacy_run(x, weight, bias, N, C, HxW, group, eps)
    xp = x.data_ptr()
    wp = xp if weight is None else weight.data_ptr()
    bp = xp if bias is None else bias.data_ptr()
    if (xp | wp | bp) & 15:
        return _legacy_run(x, weight, bias, N, C, HxW, group, eps)
    has_weight, has_bias = weight is not None, bias is not None
    key = (x.dtype, x.device.index, tuple(x.shape), N, C, HxW, group, eps,
           has_weight, has_bias, None if weight is None else weight.dtype,
           None if bias is None else bias.dtype)
    plan = _HOST_PLANS.get(key)
    if plan is False:
        return _legacy_run(x, weight, bias, N, C, HxW, group, eps)
    # Fresh independent storage for all outputs; no values or pointers cached.
    out = x.new_empty(x.shape)
    mean = x.new_empty((N, group))
    rstd = x.new_empty((N, group))
    if plan is None:
        block = _pick_block(HxW)
        cg = C // group
        even = HxW % block == 0
        try:
            ck = _gn_kernel.warmup(x,out,mean,rstd,x if weight is None else weight,
                x if bias is None else bias,eps,HW=HxW,CG=cg,G=group,
                NS=C*HxW,CS=HxW,SS=1,COUNT=cg*HxW,BLOCK=block,
                HAS_WEIGHT=has_weight,HAS_BIAS=has_bias,EVEN=even,grid=(group,N))
            launcher=ck.run
            if (getattr(launcher,'compile_only',False) or getattr(launcher,'enable_msprof_register_tensor',False)
                or getattr(launcher.metadata,'debug_enabled',False) or getattr(launcher.metadata,'debug',False)):
                raise RuntimeError('use registered profiling launcher')
            tail=(eps,HxW,cg,group,C*HxW,HxW,1,cg*HxW,block,int(has_weight),int(has_bias),int(even))
            plan=(launcher.launch,ck.function,ck.packed_metadata,tail)
            _HOST_PLANS[key]=plan
            _resolve_stream()
        except Exception:
            _HOST_PLANS[key]=False
            return _legacy_run(x,weight,bias,N,C,HxW,group,eps)
    plan[0](group,N,1,_stream_ptr(),plan[1],plan[2],None,None,None,
            xp,out.data_ptr(),mean.data_ptr(),rstd.data_ptr(),wp,bp,*plan[3])
    return out, mean, rstd


native_group_norm=run
'''
    proposal('native_group_norm','compact-contiguous-host-v3',norm,
             '连续输入缓存形状/布局分派与launch常量，使用new_empty新建三个独立输出，保留原Triton计算',
             '基于历史cProfile，减少宿主Python元数据处理与显式工厂参数成本；设备计算与数值顺序保持一致')
    print(json.dumps(dict(status='prepared',proposals=3)))


if __name__=='__main__':main()
