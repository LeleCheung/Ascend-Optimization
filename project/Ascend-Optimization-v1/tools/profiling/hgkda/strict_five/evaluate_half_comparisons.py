"""等待首个FP16提案终态后，独立评测去FP32比较扩展版本；不重发未知请求。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'
PY='/usr/local/python3.11.15/bin/python3.11'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def main():
    sys.path.insert(0,str(HERE.parent/'strict_night'))
    from admission import static
    from boundary_upgrade import terminal_workspace
    from expm1_exact_probe import qualified
    from measurements import score,computation_identity
    from signature_gate import admit
    spec=importlib.util.spec_from_file_location('strict_five_half_queue',HERE.parent/'run-nightly.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    process=BATCH/'half-comparisons-process.json';assert not process.exists(),'已有提案评测身份，不重复启动'
    identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],status='waiting_for_project_lock')
    save(process,identity)
    while True:
        prior=read(BATCH/'half-evaluator-process.json')
        if prior['status']=='completed':break
        assert prior['status'] in {'waiting_for_project_lock','running'},'先行提案异常，保留现场待核查'
        time.sleep(10)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        optimizer=read(BATCH/'optimizer-process.json');assert optimizer['status']=='completed'
        assert read(BATCH/'search-state.json')['current_job'] is None
        assert all(terminal_workspace(w) for w in (ROOT/'runs/strict-five-20261103/kg-runs').iterdir() if w.is_dir())
        q=m.Queue(BATCH.name,ROOT,'tle_yy');q.idle()
        class Host:
            def __init__(self):self.q=q
            def cmd(self,key,args,budget=2100):
                self.q.idle();return self.q.run(key,shlex.join([str(v) for v in args]),budget)
        runner=Host()
        out=BATCH/'proposals/expm1_/codex-half-comparisons-v2';source=out/'main.py'
        digest=sha(source);manual=read(out/'review.json');proposal=read(out/'spec.json')
        assert manual['status']=='reviewed_strict_triton' and manual['source_sha256']==proposal['source_sha256']==digest
        assert manual['all_branches_reviewed'] and manual['dependencies_reviewed'] and not static(source.read_text())
        leaf=BATCH/'catalog/ops/expm1_'
        signature=admit(source,leaf/'definition.json');save(out/'signature-admission.json',signature)
        assert signature['status']=='admitted'
        compiled=read(BATCH/'analysis/expm1_/compile-half-comparisons-v2/result.json')
        assert compiled['source_sha256']==digest and len(compiled['rows'])==6
        assert all(row['status']=='compiled_without_runtime' for row in compiled['rows'])
        identity.update(status='running',started_at=m.now());save(process,identity)
        result=dict(status='running',source_sha256=digest,reference='original_PyTorch_only',CANN_accepted=False)
        save(out/'result.json',result)
        try:
            normalized=computation_identity(source)
            duplicates=[]
            for formal in (ROOT/'runs/strict-five-20261103/kg-runs').glob('expm1_*/stages/optimize/work/.kernelgen/evals/round-*/result.json'):
                candidate=formal.with_name('main.py')
                if candidate.exists() and read(formal).get('status')=='PASSED' and computation_identity(candidate)==normalized:
                    duplicates.append(str(formal))
            if duplicates:
                result.update(status='duplicate_AST_not_remeasured',existing_results=duplicates)
                save(out/'result.json',result);identity.update(status='completed',finished_at=m.now());save(process,identity);return
            runtime=out/'runtime-audit.json'
            runner.cmd('half-comparisons-runtime-v2',[PY,HERE.parent/'strict_triton/runtime_audit.py','--operator','expm1_',
                '--source',source,'--leaf',leaf,'--output',runtime],1200)
            assert read(runtime)['status']=='passed'
            runner.cmd('half-comparisons-formal-v2',[PY,HERE.parent/'cann_priority/evaluate.py','--operator','expm1_',
                '--source',source,'--leaf',leaf,'--output',out/'pytorch','--server',q.server,
                '--label','strict-PyTorch','--kind','strict_Triton_Codex_proposal'],2100)
            formal=out/'pytorch/strict-PyTorch-1.result.json';value=read(formal)
            result.update(formal=str(formal),formal_status=value['status'],PyTorch=value.get('geo_mean'))
            if value['status']=='PASSED' and not value.get('is_hack'):
                extra=out/'extra-pytorch.json'
                runner.cmd('half-comparisons-extra-v2',[PY,HERE.parent/'cann_priority/validate_cann.py','--operator','expm1_',
                    '--source',source,'--leaf',leaf,'--output',extra],1200)
                checker=HERE.parent/'strict_night/expm1_exact_probe.py'
                frozen=out/('exact-expm1-checker-'+sha(checker)+'.py');assert not frozen.exists();frozen.write_bytes(checker.read_bytes())
                exact=out/'exact-expm1-probe'
                runner.cmd('half-comparisons-exact-v2',[PY,frozen,'--source',source,'--output',exact],900)
                result.update(supplement=str(extra),exact=str(exact/'result.json'))
                result['qualified_original']=read(extra)['status']=='passed' and qualified(read(exact/'result.json'),digest)
                if result['qualified_original']:
                    result['status']='evaluated';s=score(value)
                    old=read(BATCH/'starting-delivery.json')
                    old_row=next(r for r in old['rows'] if r['operator']=='expm1_')
                    if s>old_row['pytorch']['score']:
                        best=dict(source=str(source),source_sha256=digest,search_result=str(formal),search_score=s,
                            qualification='passed',reference='original_PyTorch_only',exact_expm1_probe=str(exact/'result.json'),
                            exact_expm1_checker=str(frozen),pytorch_result=str(formal),
                            provenance=proposal['attribution'],extras={'pytorch':dict(path=str(extra),status='passed')})
                        final=importlib.util.spec_from_file_location('strict_five_half_finalize',HERE.parent/'strict_night/finalize.py')
                        f=importlib.util.module_from_spec(final);final.loader.exec_module(f);f.BATCH=BATCH
                        f.verify(runner,'expm1_',best);save(out/'verified-candidate.json',best)
                        result.update(improved_over_frozen=True,backend=best['compiled_status'])
                    else:result.update(improved_over_frozen=False,reason='未超过冻结最佳，保留正式及补测结果，不重复计时')
                else:result['status']='supplement_failed'
            else:result['status']='formal_failed'
            save(out/'result.json',result)
            q.idle();identity.update(status='completed',finished_at=m.now());save(process,identity)
        except Exception as error:
            identity.update(status='failed',error=type(error).__name__+': '+str(error),finished_at=m.now());save(process,identity)
            result.update(status='requires_review',error=identity['error']);save(out/'result.json',result);raise


if __name__=='__main__':main()
