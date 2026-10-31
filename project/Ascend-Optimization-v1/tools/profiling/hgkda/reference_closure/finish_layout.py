"""沿用同一设备锁，核验首轴分块还原候选；不重复已有请求。"""
import ast
import fcntl
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run

def prepare():
    parent=run.BATCH/'proposals/cumulative_trapezoid/layout-restore-v1/main.py'
    source=parent.read_text();needle='''    _restore[(triton.cdiv(out.numel(),1024),)](
        tmp,out,tuple(int(out.shape[d]) for d in dims),
        tuple(int(out.stride(d)) for d in dims),int(out.stride(axis)),rows,n1,1024)'''
    assert source.count(needle)==1
    replacement='''    if axis == 0:
        _transpose[(-(-rows // _TILE), -(-n1 // _TILE))](
            tmp,out,(n1,),(1,),1,rows,n1,n1,rows,_TILE,_TILE)
    else:
        _restore[(triton.cdiv(out.numel(),1024),)](
            tmp,out,tuple(int(out.shape[d]) for d in dims),
            tuple(int(out.stride(d)) for d in dims),int(out.stride(axis)),rows,n1,1024)'''
    source=source.replace(needle,replacement);ast.parse(source)
    out=parent.parent.parent/'tiled-restore-v2';out.mkdir(parents=True,exist_ok=True);target=out/'main.py'
    if target.exists():assert target.read_text()==source
    else:target.write_text(source)
    proposal=dict(operator='cumulative_trapezoid',source=str(target),source_sha256=run.sha(target),parent=str(parent),parent_sha256=run.sha(parent),
        attribution='Codex修改CC第三轮：首轴采用正确参数的分块转置还原，其他轴保留通用映射',
        hypothesis='v1首轴散写还原使完整调用约0.62ms；分块连续读写减少该额外代价。默认FP16参考不稳定仍单列，不修改合同或容差。')
    run.base.save_json(out/'proposal.json',proposal);return target

def main():
    with (run.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);r=run.Runner(run.BATCH.name);source=prepare();op='cumulative_trapezoid';out=source.parent
        r.job('scan-tiled-restore-v2-formal',lambda:r.formal(op,source,out/'formal','HG-KDA-tiled-restore-v2',kind='CC_generated_Codex_tiled_layout_repair_Triton'))
        r.job('scan-tiled-restore-v2-extra',lambda:r.supplementary(op,source,out/'extra-checks.json','scan-tiled-restore-v2-extra'))
        def conditional():
            target=out/'deterministic-condition'
            r._cmd('scan-tiled-restore-v2-deterministic',[run.base.PY,HERE/'conditional_scan.py','--source',source,
                '--leaf',run.BATCH/r.q.records[op]['catalog_leaf'],'--output',target],2100)
            value=run.read(target/'deterministic-1.result.json')
            return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),passed=value.get('num_passed'),total=value.get('num_workloads'),
                output=str(target),source_sha256=run.sha(source),condition='reference_and_candidate_deterministic')
        r.job('scan-tiled-restore-v2-deterministic',conditional)
        r.state.update(status='completed_with_reference_status',current_job=None,finished_at=run.now());r.save()
        for name in ['archive_logs.py','report_v2.py','acceptance_table.py']:r._cmd('layout-final-'+name,[run.base.PY,HERE/name],180)

if __name__=='__main__':main()
