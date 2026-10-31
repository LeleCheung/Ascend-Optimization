"""接续已结束的当前任务：补齐参考符号、验证已有候选，再恢复25项队列。"""
import argparse
import fcntl
import hashlib
from pathlib import Path

from run import Runner,ROOT,HERE,PY,quote,load,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);a=p.parse_args()
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=Runner(a.batch)
        r.wait_idle()
        marker=r.q.runtime/'STOP-AFTER-JOB'
        if marker.exists():
            archived=marker.with_name('STOP-AFTER-JOB.before-followup')
            if archived.exists():archived=marker.with_name('STOP-AFTER-JOB.before-followup-'+str(__import__('time').time_ns()))
            marker.rename(archived)
            r.state.setdefault('handoffs',[]).append(dict(marker=str(archived),time=r.module.now(),reason='仅在持卡锁且服务空闲的任务边界恢复控制器'))
        r.state['status']='running'
        r.save()
        # 从已正常结束的自己的KG工作区冻结源码，不以调试成功代替正式结果。
        workspace=r.q.runtime/'kg-runs'/('negative_-'+a.batch+'-attempt-1')
        metadata=load(workspace/'.kernelgen/run-process.json')
        if not metadata.get('finished_at'):raise RuntimeError('negative旧worker未确认结束')
        generated=workspace/'stages/optimize/work/tmp/main.py'
        if generated.exists():
            frozen=r.batch/'results/negative_/handoff-evaluation/main.py'
            frozen.parent.mkdir(parents=True,exist_ok=True)
            if frozen.exists():assert frozen.read_bytes()==generated.read_bytes()
            else:frozen.write_bytes(generated.read_bytes())
            result=r.job('recover-candidate:negative_:1',lambda:r.evaluate('negative_',frozen,frozen.parent/'formal','HG-KDA-recovered'))
            if result and result['status']=='PASSED':
                gates=frozen.parent/'extra-checks.json'
                r.job('recover-extra:negative_:1',lambda:r.q.run('recover-negative-extra',quote([PY,HERE/'validate.py','--operator','negative_','--leaf',r.batch/r.q.records['negative_']['catalog_leaf'],'--source',frozen,'--output',gates]),1200))
                if gates.exists() and load(gates)['status']=='passed':
                    result['extra_checks']=str(gates);r.consider_best(r.state['operators']['negative_'],result);r.save()
        # Codex从已验证历史实现迁移的候选也走同一KGS，不把方法复用当作成绩。
        proposed=r.batch/'proposals/negative_/launch-reuse-v1/main.py'
        if proposed.exists():
            result=r.job('proposal:negative_:launch-reuse-v1',lambda:r.evaluate('negative_',proposed,proposed.parent/'formal','HG-KDA-launch-reuse'))
            if result and result.get('status')=='PASSED':
                gates=proposed.parent/'extra-checks.json'
                if not gates.exists():
                    r.job('proposal-extra:negative_:launch-reuse-v1',lambda:r.q.run('proposal-negative-extra',
                        quote([PY,HERE/'validate.py','--operator','negative_','--leaf',r.batch/r.q.records['negative_']['catalog_leaf'],
                               '--source',proposed,'--output',gates]),1200))
                if gates.exists() and load(gates)['status']=='passed':
                    result['extra_checks']=str(gates);r.consider_best(r.state['operators']['negative_'],result);r.save()
        for op in ['expm1_','native_group_norm']:
            capture=r.batch/'analysis'/op/'paired'
            if not (capture/'state.json').exists():continue
            r.job('repair-symbols:'+op+':1',lambda op=op,capture=capture:r.q.run('repair-symbols-'+op,quote([PY,HERE/'repair_capture.py','--capture',capture]),1500))
            context=capture.parent/'agent-context-v3.json'
            if not context.exists():
                r.job('reassemble:'+op+':3',lambda op=op,capture=capture,context=context:r.q.run('reassemble-'+op+'-v3',quote([PY,HERE/'assemble.py','--capture',capture,'--output',context]),600))
        # 第一轮因缓存或调度原因没交付，显式用新attempt，不覆盖已有工作区。
        for op in ['expm1_','negative_']:
            row=r.state['operators'].setdefault(op,{})
            if (row.get('best') or {}).get('geo_mean',0)>.8:continue
            row['status']='HG-KDA接续优化';r.save()
            result=r.job('search-retry:'+op+':2',lambda op=op:r.search(op,attempt=2))
            if result and result.get('status')=='PASSED':
                r.consider_best(row,result);r.save()
        save_json(r.batch/'followup-ready.json',dict(status='resuming_queue',time=r.module.now()))
        r.execute()


if __name__=='__main__':
    try:main()
    except Exception as error:
        # 原任务和请求保持原样；失败原因同步输出日志，避免tmux静默退出。
        print('FOLLOWUP_FAILED '+type(error).__name__+': '+str(error),flush=True)
        raise
