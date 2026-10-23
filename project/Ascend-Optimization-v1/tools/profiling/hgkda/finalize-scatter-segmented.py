#!/usr/bin/env python3
"""已有真实达标候选后，完成原父版反例、路径采集和交付，不再重复搜索。"""
import argparse
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import time

TOOLS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scatter_validated_delivery',TOOLS/'run-scatter-segmented-followup.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=module.base


def execute(name):
    batch=module.PROJECT/'operators/campaigns'/name
    main=module.PROJECT/'operators/campaigns'/module.MAIN
    # 仅接管尚未开始、已导入真实完整结果的排队批次。
    state=base.read(batch/'endurance-state.json')
    assert state['status']=='created' and 'started_epoch' not in state
    selected=state['operators']['scatter_add']['best']
    assert selected['source_sha256']=='72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e'
    assert selected['geo_mean']>.8
    assert base.sha(Path(selected['source']))==selected['source_sha256']
    deadline=time.monotonic()+24*3600
    waiter=batch/'wait-state.json'
    with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        while True:
            parent=base.read(main/'endurance-state.json')
            post=base.read(main/'postflight-state.json') if (main/'postflight-state.json').exists() else {}
            if parent['status']=='completed' and post.get('status') in ['passed','needs_attention']:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:pass
            module.save(waiter,dict(status='waiting_for_main_and_postflight',main=parent['status'],postflight=post.get('status')))
            if time.monotonic()>deadline:raise TimeoutError('排队预算结束；未提交设备请求')
            time.sleep(15)
        campaign=module.Campaign(name)
        try:
            module.save(waiter,dict(status='running',mode='validated_candidate_delivery'))
            campaign.step('baseline-version-check',lambda:campaign.run('verify-baseline',[
                base.PY,TOOLS/'verify-baseline.py','--server',campaign.q.server,'--output',batch/'baseline-verified.json'],600))
            assert campaign.state['jobs']['baseline-version-check']['status']=='completed'
            parent=campaign.step('parent:scatter_add',lambda:campaign.baseline('scatter_add'))
            candidate=campaign.step('validate:scatter_add:delivery-reuse:'+selected['source_sha256'][:12],
                lambda:campaign.candidate('scatter_add',Path(selected['source']),Path(selected['source']).parent,'delivery-reuse'))
            assert candidate and candidate['status']=='passed'
            op=campaign.state['operators']['scatter_add']
            op.update(parent=parent,best=candidate,status='有效最佳已交付',
                search_decision='已有通过完整正式、补充数值及动态范围门禁的1.900×源码；停止重复模型搜索，完成交付采集')
            campaign.save()
            campaign.step('observe-parent:scatter_add',lambda:campaign.observe('scatter_add',batch/'seeds/scatter_add.py',batch/'research/scatter_add/parent'))
            campaign.step('observe-final:scatter_add',lambda:campaign.observe('scatter_add',Path(candidate['source']),batch/'research/scatter_add/final'))
            campaign.step('metrics-final:scatter_add',lambda:campaign.metrics('scatter_add',candidate,batch/'research/scatter_add/final-metrics'))
            campaign.finish()
            result=module.audit(batch)
            module.save(batch/'completion-audit.json',result)
            module.save(waiter,dict(status='passed' if result['status']=='passed' and result.get('meets_target') else 'needs_attention',audit=result))
        except Exception as error:
            campaign.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));campaign.save()
            module.save(waiter,dict(status='needs_attention',error=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    execute(parser.parse_args().batch)
