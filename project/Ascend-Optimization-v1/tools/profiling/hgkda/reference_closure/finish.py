"""首轮KG完成后继续命名NPU对照；每步沿用同一项目锁和状态。"""
import fcntl
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run

def main():
    with (run.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);r=run.Runner(run.BATCH.name);r.state['status']='running';r.save()
        parent=run.PROJECT/'operators/campaigns/acceptance-three-20261026/npu-decomposition-catalog-v2/ops'
        for op in ['fractional_max_pool3d','_conj_physical']:
            leaf=r.audited_alternate(op,parent/op/'oracle.py','NPU-decomposition-v2');source=r.starting_source(op)
            out=run.BATCH/'comparisons'/op/'NPU-decomposition-v2'
            r.job('supplemental-full-v2:'+op,lambda op=op,source=source,out=out,leaf=leaf:r.formal(op,source,out,'HG-KDA-vs-NPU-v2',leaf=leaf,kind='existing_candidate_vs_named_NPU_decomposition'))
            gates=out/'extra-checks.json'
            def checks(op=op,source=source,leaf=leaf,gates=gates):
                if not gates.exists():
                    try:r._cmd('supplemental-extra-v2-'+op,[run.base.PY,HERE/'validate_reference.py','--operator',op,'--source',source,'--leaf',leaf,'--output',gates],1200)
                    except RuntimeError:
                        if not gates.exists():raise
                value=run.read(gates)
                from validation_status import supplement_status
                return dict(status=supplement_status(op,value,run.sha(source)),passed=value['passed'],total=value['total'],extra_checks=str(gates))
            r.job('supplemental-extra-v2:'+op,checks)
        r.state.update(status='completed_with_reference_status',current_job=None,finished_at=run.now());r.save()
        r._cmd('archive-controller-logs',[run.base.PY,HERE/'archive_logs.py'],120)
        r._cmd('delivery-report-v2',[run.base.PY,HERE/'report_v2.py'],180)

if __name__=='__main__':main()
