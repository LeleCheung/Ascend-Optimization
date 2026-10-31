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
        def inverse_audit():
            op='linalg_inv_ex';reference=run.read(run.BATCH/'reference-index.json')[op]
            out=run.BATCH/'references'/op/'isolated-audit-v1'
            r._cmd('cann-inverse-isolated',[run.base.PY,HERE/'probe_inverse_isolated.py','--source',reference['source'],
                '--leaf',run.BATCH/r.q.records[op]['catalog_leaf'],'--output',out],1800)
            return run.read(out/'summary.json')
        r.job('cann-inverse-isolated-v1',inverse_audit)
        for reference_op,revision in [('fake_quantize_per_tensor_affine_cachemask_out','v2'),('batch_norm_with_update_functional','v3')]:
            reference=run.read(run.BATCH/('reference-index-'+revision+'.json'))[reference_op]
            source=Path(reference['source']);output=source.parent/'dynamic-semantics.json'
            def semantic_check(op=reference_op,source=source,output=output):
                if not output.exists():r._cmd('cann-dynamic-'+op,[run.base.PY,HERE/'check_cann_semantics.py','--operator',op,'--source',source,
                    '--leaf',run.BATCH/r.q.records[op]['catalog_leaf'],'--output',output],900)
                return run.read(output)
            r.job('cann-dynamic-semantics:'+reference_op,semantic_check)
        source=run.BATCH/'proposals/linalg_inv_ex/padded-block-v1/main.py'
        if source.exists():
            op='linalg_inv_ex';out=source.parent;leaf=run.BATCH/r.q.records[op]['catalog_leaf'];probe=out/'diagnostic.json'
            def padded_probe():
                if not probe.exists():r._cmd('inverse-padded-diagnostic',[run.base.PY,HERE/'probe_candidate.py','--source',source,'--leaf',leaf,'--output',probe],900)
                return run.read(probe)
            result=r.job('inverse-padded-diagnostic',padded_probe)
            r.job('inverse-padded-formal',lambda:r.formal(op,source,out/'formal','HG-KDA-padded-block',kind='Codex_modified_CC_Triton_candidate'))
            if result and result.get('status')=='diagnostic_passed':
                r.job('inverse-padded-extra',lambda:r.supplementary(op,source,out/'extra-checks.json','inverse-padded-extra'))
        # fix已有合格厂商参考；直接更换命名oracle同合同评测，不相除两批汇总分。
        op='fix';reference=Path(run.read(run.BATCH/'reference-index.json')[op]['source'])
        leaf=r.alternate(op,reference,'CANN-direct-qualified');source=r.starting_source(op);out=run.BATCH/'comparisons'/op/'CANN-direct'
        r.job('candidate-vs-cann:fix',lambda:r.formal(op,source,out,'HG-KDA-vs-CANN',leaf=leaf,kind='AscendC_candidate_vs_CANN'))
        r.job('candidate-extra-vs-cann:fix',lambda:r.supplemental_at(op,source,out/'extra-checks.json','candidate-extra-vs-cann-fix',leaf))
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
        r._cmd('acceptance-table',[run.base.PY,HERE/'acceptance_table.py'],180)

if __name__=='__main__':main()
