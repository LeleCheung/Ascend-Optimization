#!/usr/bin/env python3
"""等待整批后置任务终态，再保留原始失败地汇总语义与额外修复证据。"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--container',default='tle_yy')
    parser.add_argument('--wait-hours',type=float,default=12.)
    args=parser.parse_args()
    if not args.batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    project=args.root/'Ascend-Optimization/project/Ascend-Optimization-v1'
    tools=project/'tools/profiling/hgkda';batches=project/'operators/campaigns'
    directory=batches/args.batch;output=directory/'postflight.json'
    repairs=[args.batch+'-'+s for s in ['value-stride-repair','value-profile-repair','value-hg-repair','value-hg-profile-guided']]
    waiting=[args.batch,args.batch+'-adapter-repairs',args.batch+'-silu-adapter-repair']+repairs
    state={'status':'waiting','batch':args.batch,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        state['updated_at']=datetime.datetime.utcnow().isoformat()+'Z'
        temporary=output.with_suffix('.tmp');temporary.write_bytes((json.dumps(state,ensure_ascii=False,indent=2)+'\n').encode());temporary.replace(output)
    started=time.monotonic()
    while True:
        statuses={}
        for batch in waiting:
            path=batches/batch/'queue-state.json'
            statuses[batch]=json.loads(path.read_text()).get('status') if path.exists() else 'queued'
        state['queue_states']=statuses;save()
        if all(s in ['completed','needs_attention'] for s in statuses.values()):break
        if time.monotonic()-started>args.wait_hours*3600:
            state.update(status='needs_attention',error='后置汇总等待超预算；保留原进程，不重复提交设备请求');save();return 1
        time.sleep(30)
    state['status']='checking';save()
    log=args.root/'runs'/args.batch/'postflight-checks.log'
    with log.open('ab') as stream:
        stream.write(('\nPOSTFLIGHT '+state['updated_at']+'\n').encode());stream.flush()
        # finalize自持本项目卡锁；同源码/检查器身份的失败证据直接保留，不重跑覆盖。
        final=subprocess.run(['python3',str(tools/'finalize-nightly.py'),'--root',str(args.root),
            '--batch',args.batch,'--container',args.container],stdout=stream,stderr=subprocess.STDOUT)
        state['original_finalizer_exit_code']=final.returncode;save()
        command=['docker','exec',args.container,'/usr/local/python3.11.15/bin/python3.11',str(tools/'verify-semantic-repairs.py')]
        for batch in repairs:command+=['--batch-dir',str(batches/batch)]
        command+=['--output',str(directory/'repair-verification.json')]
        checked=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        state['repair_verifier_exit_code']=checked.returncode
    state['status']='checks_finished' if checked.returncode==0 else 'needs_attention'
    state['scope']='原始组严格门禁、额外修复有效性分别报告；checks_finished不等于全部候选正确或goal完成'
    save();print(json.dumps(state,ensure_ascii=False));return int(checked.returncode!=0)


if __name__=='__main__':raise SystemExit(main())
