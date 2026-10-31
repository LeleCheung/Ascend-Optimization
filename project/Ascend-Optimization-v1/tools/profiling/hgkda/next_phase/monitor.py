"""只读夜间进展：工作区终态、当前任务和已完成结果，不读取模型思考内容。"""
import argparse
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def snapshot(batch, runtime):
    state=read(batch/'next-state.json')
    jobs=state.get('jobs',{})
    summary=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status=state.get('status'),current_job=state.get('current_job'),error=state.get('error'),
        jobs={kind:sum(row.get('status')==kind for row in jobs.values())
              for kind in ['running','completed','failed']},operators=[],workers=[])
    for op,row in state.get('operators',{}).items():
        best=row.get('best') or {};base=row.get('baseline') or {}
        summary['operators'].append(dict(operator=op,status=row.get('status'),
            flaggems=dict(status=base.get('status'),passed=base.get('num_passed'),total=base.get('num_workloads'),speedup=base.get('geo_mean')),
            current=dict(status=best.get('status'),passed=best.get('num_passed'),total=best.get('num_workloads'),speedup=best.get('geo_mean')),
            extra_checks=best.get('extra_checks')))
    for metadata in sorted((runtime/'kg-runs').glob('*/.kernelgen/run-process.json')):
        process=read(metadata);workspace=metadata.parent.parent;progress=workspace/'.kernelgen/run-progress.json'
        stages=read(progress) if progress.exists() else {}
        rounds=[]
        for result in sorted(workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/result.json')):
            data=read(result);rounds.append(dict(round=result.parent.name,status=data.get('status'),
                passed=data.get('num_passed'),total=data.get('num_workloads'),speedup=data.get('geo_mean')))
        summary['workers'].append(dict(workspace=workspace.name,state=process.get('state'),
            pid=process.get('pid'),started_at=process.get('started_at'),finished_at=process.get('finished_at'),
            stage=stages.get('stage'),progress_updated_at=stages.get('updated_at'),rounds=rounds))
    summary['failed_jobs']=[dict(job=key,error=row.get('error')) for key,row in jobs.items() if row.get('status')=='failed']
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(snapshot(args.batch,args.runtime),ensure_ascii=False,indent=2))
