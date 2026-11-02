"""在独立输出目录完成一次新KGS复测；不覆盖本批证据，不重复未知请求。"""
import argparse
from pathlib import Path
from common import BATCH,HERE,PREVIOUS,ROOT,read,save,sha,load
from gate import admit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',required=True)
    p.add_argument('--baseline',choices=['CANN','PyTorch'],default='CANN')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import fcntl
    output=a.output.resolve()
    if output.exists():raise FileExistsError('输出目录已存在；先检查已有请求和结果，禁止盲目重发：'+str(output))
    if output==BATCH.resolve() or BATCH.resolve() in output.parents:raise ValueError('新复测输出须放独立目录，不能写入冻结批次')
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=load('strict_replay_runner',HERE/'run.py').Runner()
        record=runner.index[a.operator];source=Path(record['source']);admit(source,source.parent/'review.json')
        leaf=BATCH/runner.q.records[a.operator]['catalog_leaf']
        if a.baseline=='CANN':
            previous=next(r for r in read(PREVIOUS/'delivery.json')['rows'] if r['operator']==a.operator)
            if not previous.get('cann_reference_qualified'):raise ValueError('该算子尚无合格CANN参考，不能以其他参考代替')
            reference=Path(previous['reference_source']);assert sha(reference)==previous['reference_source_sha256']
            leaf=BATCH/'cann-catalog/ops'/a.operator
        output.mkdir(parents=True)
        runner.q.runtime=output/'controller-logs';runner.q.runtime.mkdir()
        runner.wait()
        lock_software=read(BATCH/'software-lock.json')['service_software']
        import json
        current=json.load(runner.q.opener.open(runner.q.server+'/status',timeout=20))
        assert current['software']==lock_software and current['timing']=='walltime'
        save(output/'replay-spec.json',dict(operator=a.operator,baseline=a.baseline,source=str(source),source_sha256=sha(source),
            contract={name:sha(leaf/name) for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
            software=current['software'],measurement='new_once',scope='一次新完整评测；不改原合同、不平均历史成绩、不替代已有补测和后端审查'))
        result=runner.formal(a.operator,source,leaf,output/'formal','replay-'+a.baseline)
        runner.wait();save(output/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['status']!='PASSED' or result.get('is_hack'):raise SystemExit(1)


if __name__=='__main__':main()
