"""只对新增过线源码做一次稳定性确认；不重选源码，不重复测试历史12项。"""
from pathlib import Path
import json


def process(r):
    from run import PY,quote
    unfinished=[key for key,value in r.state.get('jobs',{}).items()
                if value.get('status')=='running' and not key.startswith('final-confirm:')]
    if unfinished:
        raise r.module.Attention('存在尚未确认结束的任务，不能收尾 '+str(unfinished))
    for op in r.q.plan['priority_operators']:
        row=r.state['operators'].get(op,{})
        best=row.get('best') or {}
        initial=r.state['jobs'].get('verify-current:'+op,{}).get('result') or {}
        if (best.get('status')!='PASSED' or best.get('supplementary_status')!='passed'
                or best.get('geo_mean',0)<=.8 or best.get('source_sha256')==initial.get('source_sha256')):
            continue
        source=Path(best['source']);sha=best['source_sha256']
        out=r.batch/'final-confirmation'/op/sha[:12]
        if op=='cross_out':
            safety=out/'cross-safety.json'
            checker=Path(__file__).resolve().parent/'check_cross.py'
            r.job('cross-safety:'+sha[:12],lambda:r.q.run('cross-safety-'+sha[:12],
                quote([PY,checker,'--source',source,'--output',safety]),1200))
            if not safety.exists() or json.loads(safety.read_text()).get('status')!='passed':
                raise r.module.Attention('最终叉积未通过动态dim/共享storage追加门禁，保留原件待修复')
            best['cross_safety']=str(safety)
        result=r.job('final-confirm:'+op+':'+sha[:12],
                     lambda op=op,source=source,out=out:r.evaluate(op,source,out,'HG-KDA-final-confirm'))
        if not result:raise r.module.Attention('最终确认未产生结果 '+op)
        result.update(extra_checks=best['extra_checks'],supplementary_status=best['supplementary_status'],
                      attribution='同一冻结源码的完整KGS稳定性确认；沿用同SHA已通过补测；不计第二项优化')
        if best.get('cross_safety'):result['cross_safety']=best['cross_safety']
        row['final_confirmation']=result
        row['status']=('两次完整评测及补测通过，超过0.8×' if
                       result.get('status')=='PASSED' and result.get('geo_mean',0)>.8
                       else '最终确认未稳定超过0.8×，保留原始结果')
        r.save()
    rejected=r.batch/'disqualified-sources.json'
    if rejected.exists():
        for sha,entry in json.loads(rejected.read_text()).items():
            if 'round-0005' not in entry['source']:continue
            out=r.batch/'source-review'/sha[:12]/'cross-safety-negative-control.json'
            checker=Path(__file__).resolve().parent/'check_cross.py'
            r.job('cross-rejected-negative-control:'+sha[:12],lambda:r.q.run('cross-negative-control-'+sha[:12],
                quote([PY,checker,'--source',entry['source'],'--output',out,'--allow-failure']),1200))
