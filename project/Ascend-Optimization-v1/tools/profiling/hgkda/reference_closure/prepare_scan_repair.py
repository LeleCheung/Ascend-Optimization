"""对CC第三轮累计积分候选修复通用布局还原，不改原合同。"""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'

def main():
    parent=ROOT/'runs/reference-closure-20261031/kg-runs/cumulative_trapezoid-reference-closure-20261031-attempt-1/stages/optimize/work/.kernelgen/evals/round-0003/main.py'
    source=parent.read_text();needle='    _transpose[(-(-rows // _TILE), -(-n1 // _TILE))](\n        tmp, out, (rows,), (n1,), 1, rows, n1, 1, rows, _TILE, _TILE)'
    assert source.count(needle)==1
    helper='''
@triton.jit
def _restore(SRC, DST, RSHAPE: tl.constexpr, OSTRIDES: tl.constexpr,
             AXIS_STRIDE: tl.constexpr, ROWS: tl.constexpr, N1: tl.constexpr,
             BLOCK: tl.constexpr):
    linear=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=linear//N1
    coordinate=linear%N1
    remainder=row
    address=coordinate*AXIS_STRIDE
    for d in tl.static_range(len(RSHAPE)-1,-1,-1):
        address+=(remainder%RSHAPE[d])*OSTRIDES[d]
        remainder=remainder//RSHAPE[d]
    value=tl.load(SRC+linear,linear<ROWS*N1,other=0)
    tl.store(DST+address,value,linear<ROWS*N1)

_restore=_Cached(_restore)

'''
    source=source.replace('def _pow2(n):',helper+'def _pow2(n):')
    source=source.replace(needle,'''    _restore[(triton.cdiv(out.numel(),1024),)](
        tmp,out,tuple(int(out.shape[d]) for d in dims),
        tuple(int(out.stride(d)) for d in dims),int(out.stride(axis)),rows,n1,1024)''')
    source=source.replace('if low and n1 >= _SCRATCH_MIN:','if low and _SCRATCH_MIN <= n1 <= block:')
    ast.parse(source);out=BATCH/'proposals/cumulative_trapezoid/layout-restore-v1';out.mkdir(parents=True,exist_ok=True)
    target=out/'main.py'
    if target.exists():assert target.read_text()==source
    else:target.write_text(source)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proposal=dict(operator='cumulative_trapezoid',parent=str(parent),parent_sha256=sha(parent),source_sha256=sha(target),
        attribution='CC第三轮生成＋Codex通用布局修复',
        hypothesis='将打包扫描向量按输出shape与stride还原，修复把向量编号当作扫描坐标的错误；单块scratch仅用于长度不超过block的情形。',
        scope='通用rank／axis／stride映射；合同、容差和参考不变；完整KGS与补测前不宣称通过。')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    print(json.dumps(proposal,ensure_ascii=False))

if __name__=='__main__':main()
