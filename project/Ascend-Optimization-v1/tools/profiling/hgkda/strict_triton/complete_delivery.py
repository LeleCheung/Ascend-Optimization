"""排在编译核验之后补齐三项非空trace、最终审计与简报；持卡锁串行。"""
import fcntl
from pathlib import Path
import shutil
from common import BATCH,HERE,NEXT,ROOT,PY,read,save,sha,load


def main():
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert read(BATCH/'compiled-verification.json')['status']=='completed'
        assert read(BATCH/'state.json')['status']=='completed'
        runner=load('strict_completion_runner',HERE/'run.py').Runner()
        state=read(BATCH/'state.json')
        progress=BATCH/'completion-state.json';save(progress,dict(status='running'))
        parser=load('strict_final_trace_parser',HERE.parent/'threeway/trace_summary.py')
        for op in ['selu_','rad2deg_','log10_']:
            row=state['operators'][op];source=Path(row['source']);directory=source.parent/'device-path-nonempty-v2'
            directory.mkdir(parents=True,exist_ok=True)
            compiled_spec=source.parent/'actual-compiled-v2/spec.json';spec=directory/'spec.json'
            if spec.exists():assert spec.read_bytes()==compiled_spec.read_bytes()
            else:spec.write_bytes(compiled_spec.read_bytes())
            trace=directory/'trace.json'
            if not trace.exists():
                runner.wait();runner.cmd('strict-trace-nonempty-'+op,[PY,NEXT/'driver.py','--spec',spec,'--side','candidate','--trace',trace],600)
            summary=parser.summarize(trace);assert summary['status']=='observed',op
            save(directory/'summary.json',summary)
            archive=BATCH/'revisions'/op/'empty-trace-v1-state.json'
            if not archive.exists():save(archive,row.get('device_path',{}))
            row['device_path']=dict(status='captured',trace=str(trace),trace_sha256=sha(trace),summary=str(directory/'summary.json'),case=read(spec)['case']['name'])
            save(BATCH/'state.json',state)
            print('NONEMPTY_TRACE_VERIFIED',op,summary['kernel_names'],flush=True)
        runner.wait()
        load('strict_complete_history',HERE/'audit_history.py').main()
        load('strict_complete_audit',HERE/'audit_delivery.py').main()
        load('strict_complete_summary',HERE/'summarize.py').main()
        logs=BATCH/'controller-logs';logs.mkdir(exist_ok=True)
        for path in (ROOT/'runs'/BATCH.name).glob('*.log'):shutil.copyfile(path,logs/path.name)
        save(progress,dict(status='completed',accepted=read(BATCH/'delivery.json')['accepted'],
            audit_sha256=sha(BATCH/'delivery-audit.json'),summary_sha256=sha(BATCH/'result-summary.json')))
        print('STRICT_DELIVERY_COMPLETED',read(BATCH/'delivery.json')['accepted'],flush=True)


if __name__=='__main__':main()
