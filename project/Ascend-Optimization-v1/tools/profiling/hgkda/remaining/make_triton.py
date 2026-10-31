"""保留数学语义，迁移单缓冲多块处理及编译期常量。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def proposal(op,name,body,summary,prediction):
    import ast
    ast.parse(body)
    folder=BATCH/'proposals'/op/name;folder.mkdir(parents=True,exist_ok=True)
    source=folder/'main.py';raw=body.encode('utf8')
    if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
    source.write_bytes(raw)
    (folder/'proposal.json').write_text(json.dumps(dict(operator=op,id=name,source='main.py',
        source_sha256=hashlib.sha256(raw).hexdigest(),summary=summary,prediction=prediction,
        attribution='Codex基于上轮有效方法和当前源码编写；固定KGS及语义补测验收'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    get=lambda op:(BATCH/'starting-points'/(op+'.py')).read_text(encoding='utf8')
    exp=get('expm1_')
    # 同一准确公式；大输入取消跨全部张量的长循环，每个program只算两个块。
    begin=exp.index('@triton.jit\ndef _expm1_streaming(')
    end=exp.index('@triton.jit\ndef _expm1_scaled(',begin)
    exp=exp[:begin]+'''@triton.jit
def _expm1_streaming(a_ptr,n: tl.constexpr,BLOCK: tl.constexpr):
    lane=tl.arange(0,BLOCK)
    for part in tl.static_range(2):
        offset=tl.program_id(0)*(2*BLOCK)+part*BLOCK+lane
        x=tl.load(a_ptr+offset,offset<n,other=0.0)
        y=_expm1_value(x)
        tl.store(a_ptr+offset,y.to(a_ptr.dtype.element_ty),offset<n)


'''+exp[end:]
    exp=exp.replace('grid = min(40, grid)','grid = (n + 2*block - 1) // (2*block)')
    exp=exp.replace('kernel.warmup(A, n, BLOCK=block, grid=(1,))','kernel.warmup(A, n, BLOCK=block, grid=(1,),num_stages=1,multibuffer=False)')
    proposal('expm1_','two-tiles-single-buffer-v1',exp,
             '保留小值相对精度及零符号公式；大输入每program处理两个4096块，关闭多缓冲',
             '减少持久长循环与多缓冲压力，同时让更多program分担大输入；小值精度补测保留')

    fmod=get('fmod_tensor_')
    begin=fmod.index('@triton.jit\ndef _fmod_flat(');end=fmod.index('@triton.jit\ndef _fmod_strided(',begin)
    fmod=fmod[:begin]+'''@triton.jit
def _fmod_flat(A,B,N: tl.constexpr,BLOCK: tl.constexpr):
    lane=tl.arange(0,BLOCK)
    for part in tl.static_range(2):
        offset=tl.program_id(0)*(2*BLOCK)+part*BLOCK+lane
        a=tl.load(A+offset,offset<N,other=0)
        b=tl.load(B+offset,offset<N,other=1)
        tl.store(A+offset,_fmod_value(a,b),offset<N)


'''+fmod[end:]
    fmod=fmod.replace('(n + _BLOCK - 1) // _BLOCK','(n + 2*_BLOCK - 1) // (2*_BLOCK)')
    fmod=fmod.replace('grid=(1,), enable_fp_fusion=False','grid=(1,), enable_fp_fusion=False,num_stages=1,multibuffer=False')
    fmod=fmod.replace('key = (A.dtype, A.get_device(), n)','key = (A.dtype, B.dtype, A.get_device(), n)')
    proposal('fmod_tensor_','two-tiles-single-buffer-v1',fmod,
             '保留截断余数及原机零符号数值顺序；两个4096块合为一program，关闭多缓冲',
             '减少program与调度开销；广播、双方stride、极值补测继续约束')

    inverse=get('linalg_inv_ex')
    # 原参考在CPU采用带融合乘加的LU；用准确除法和显式FMA检查重复行近奇异差异。
    inverse=inverse.replace('col_vals / safe_pivot','tl.div_rn(col_vals, safe_pivot)')
    inverse=inverse.replace('work - multipliers[:, None] * u_row[None, :]','tl.fma(-multipliers[:, None],u_row[None, :],work)')
    inverse=inverse.replace('rhs_row_j / safe_diag','tl.div_rn(rhs_row_j, safe_diag)')
    inverse=inverse.replace('rhs - l_col[:, None] * rhs_row_j[None, :]','tl.fma(-l_col[:, None],rhs_row_j[None, :],rhs)')
    inverse=inverse.replace('rhs - u_col[:, None] * scaled_row[None, :]','tl.fma(-u_col[:, None],scaled_row[None, :],rhs)')
    proposal('linalg_inv_ex','lu-accurate-div-fma-v1',inverse,
             'LU与回代改为正确舍入除法及显式融合乘加；不改参考、奇异输入、容差和info',
             '检查重复行矩阵在参考CPU LU舍入残差下表现为近奇异的差异；能否匹配须正式评测确认')
    inverse2=inverse.replace('tl.div_rn(col_vals, safe_pivot)','col_vals * tl.div_rn(1.0, safe_pivot)')
    proposal('linalg_inv_ex','lu-reciprocal-scale-fma-v2',inverse2,
             '按CPU BLAS缩放形式先求FP32主元倒数再乘列，尾部更新显式FMA；保持原输入和容差',
             '直接除法对重复行给出精确1，倒数乘法可能产生参考LU的非零舍入残差；这是待验证的数值路径假设')
    print('Prepared 4 Triton proposals')


if __name__=='__main__':main()
