"""累计积分混合实现：非末轴由Triton增量＋直接CANN扫描，完整验证条件分列。"""
import ast
import fcntl
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run

def prepare():
    build=run.read(run.BATCH/'cann-cumsum-build.json');assert run.sha(build['binary'])==build['binary_sha256']
    parent=run.BATCH/'proposals/cumulative_trapezoid/tiled-restore-v2/main.py';source=parent.read_text()
    incremental='''
@triton.jit
def _increments_axis(Y,Z,SHAPE:tl.constexpr,STRIDES:tl.constexpr,AXIS:tl.constexpr,
                     TOTAL:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    linear=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    remain=linear;address=tl.zeros((BLOCK,),tl.int64)
    for d in tl.static_range(len(SHAPE)-1,-1,-1):
        address+=(remain%SHAPE[d])*STRIDES[d]
        remain=remain//SHAPE[d]
    left=tl.load(Y+address,linear<TOTAL,other=0).to(tl.float32)
    right=tl.load(Y+address+STRIDES[AXIS],linear<TOTAL,other=0).to(tl.float32)
    summed=(left+right).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Z+linear,(summed*SCALE).to(Z.dtype.element_ty),linear<TOTAL)
_increments_axis=_Cached(_increments_axis)
'''
    bridge='''
import hashlib as _hashlib
import importlib.util as _importlib
from pathlib import Path as _Path
_binary=_Path(%r)
assert _hashlib.sha256(_binary.read_bytes()).hexdigest()==%r
_spec=_importlib.spec_from_file_location(%r,_binary)
_cann_scan=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_cann_scan)
'''%(build['binary'],build['binary_sha256'],build['module_name'])
    source=source.replace('def _pow2(n):',incremental+bridge+'\ndef _pow2(n):')
    needle='    buf = torch.empty(rows * (n + n1), dtype=y.dtype, device=y.device)'
    assert source.count(needle)==1
    source=source[:source.index(needle)]+'''    increments=torch.empty(tuple(shape),dtype=y.dtype,device=y.device)
    _increments_axis[(triton.cdiv(increments.numel(),2048),)](
        y,increments,tuple(shape),tuple(y.stride()),axis,increments.numel(),scale,2048)
    return _cann_scan.cumsum(increments,axis)
'''
    ast.parse(source);out=parent.parent.parent/'hybrid-cann-v3';out.mkdir(parents=True,exist_ok=True);target=out/'main.py'
    if target.exists():assert target.read_text()==source
    else:target.write_text(source)
    run.base.save_json(out/'proposal.json',dict(operator='cumulative_trapezoid',source=str(target),source_sha256=run.sha(target),parent=str(parent),parent_sha256=run.sha(parent),
        cann_build=build,attribution='CC末轴扫描候选＋Codex非末轴Triton增量与CANN C++扫描桥',
        scope='混合实现，厂商只承担扫描步骤；新数据一致性和确定性条件分别检查；不作为纯Triton成绩。'))
    return target

def main():
    with (run.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);r=run.Runner(run.BATCH.name);source=prepare();op='cumulative_trapezoid';out=source.parent
        r.job('scan-hybrid-v3-formal',lambda:r.formal(op,source,out/'formal','HG-KDA-hybrid-cann-v3',kind='CC_Triton_Codex_increment_CANN_scan_hybrid'))
        r.job('scan-hybrid-v3-extra',lambda:r.supplementary(op,source,out/'extra-checks.json','scan-hybrid-v3-extra'))
        def conditional():
            target=out/'deterministic-condition'
            r._cmd('scan-hybrid-v3-deterministic',[run.base.PY,HERE/'conditional_scan.py','--source',source,'--leaf',run.BATCH/r.q.records[op]['catalog_leaf'],'--output',target],2100)
            value=run.read(target/'deterministic-1.result.json')
            return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),passed=value.get('num_passed'),total=value.get('num_workloads'),output=str(target),source_sha256=run.sha(source),condition='reference_and_candidate_deterministic')
        r.job('scan-hybrid-v3-deterministic',conditional)
        r.state.update(status='completed_with_reference_status',current_job=None,finished_at=run.now());r.save()
        for name in ['archive_logs.py','report_v2.py','acceptance_table.py']:r._cmd('hybrid-final-'+name,[run.base.PY,HERE/name],180)

if __name__=='__main__':main()
