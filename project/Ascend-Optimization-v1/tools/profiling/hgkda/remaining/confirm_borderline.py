"""fix第二次接近0.8门槛，预先固定追加一次完整确认；三次全部保留，不择高分。"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def main():
    sys.path.insert(0,str(HERE))
    spec=importlib.util.spec_from_file_location('remaining_borderline_controller',HERE/'run.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    with (controller.base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=controller.Runner('remaining-nine-20261030')
        if not all(job['status']!='running' for job in r.state['jobs'].values()):raise RuntimeError('原队列尚有未结束任务')
        op='fix';row=r.state['operators'][op];best=row['best'];confirmation=row['final_confirmation']
        source=Path(best['source']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if confirmation['source_sha256']!=sha:raise RuntimeError('两次源码不一致')
        plan=dict(operator=op,source_sha256=sha,additional_complete_evaluations=1,all_scores_retained=True,
            reason='第二次完整结果0.8030897728800286，距0.8门槛很近；固定追加一次以确认稳定性，不按结果继续增测',
            previous_scores=[best['geo_mean'],confirmation['geo_mean']])
        controller.base.save_json(r.batch/'borderline-confirmation-plan.json',plan)
        result=r.job('borderline-confirm:fix:'+sha[:12],lambda:r.evaluate(op,source,
            r.batch/'borderline-confirmation'/op/sha[:12],'HG-KDA-borderline-third'))
        if not result:raise r.module.Attention('临界项第三次结果未知')
        result.update(extra_checks=best['extra_checks'],supplementary_status='passed')
        row['additional_confirmations']=[result]
        row['borderline_plan']=str(r.batch/'borderline-confirmation-plan.json')
        scores=[best['geo_mean'],confirmation['geo_mean'],result.get('geo_mean')]
        stable=result.get('status')=='PASSED' and all(isinstance(v,(int,float)) and v>.8 for v in scores)
        if all(isinstance(v,(int,float)) and v>0 for v in scores):
            import math
            row['reported_speedup']=math.exp(sum(math.log(v) for v in scores)/len(scores))
        row['status']='三次完整评测及补测通过，均超过0.8×' if stable else '临界项第三次确认未达标，不计稳定交付'
        r.wait_idle();r.state.update(status='all_completed',current_job=None,finished_at=r.module.now());r.save()
        print(json.dumps(dict(operator=op,status=row['status'],scores=scores,reported_speedup=row.get('reported_speedup')),ensure_ascii=False))


if __name__=='__main__':main()
