"""任务边界加载新控制器，验证修正候选、补采参考后继续同一25项队列。"""
import argparse
import fcntl
import hashlib
import re
import time
from pathlib import Path

from run import Runner,ROOT,HERE,TOOLS,PY,quote,load,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    parser.add_argument('--handoff-token',required=True)
    parser.add_argument('--negative-proposal',default='launch-reuse-bitwise-v2')
    parser.add_argument('--host-profile',action='append',default=[],help='对已通过完整评测及补测的算子分解宿主调用')
    parser.add_argument('--paired-best',action='append',default=[],help='对当前已验证最佳版本补采最大冻结输入的真机指标和IR')
    parser.add_argument('--paired-record',action='append',default=[],help='对指定已验证job记录补采，不更改最佳版本或重跑正式评测')
    parser.add_argument('--reference-audit',action='append',default=[],help='提前审计指定算子的代表PyTorch设备路径，复用原队列审计记录')
    parser.add_argument('--skip-negative-refresh',action='store_true',help='仅更新控制器时，保留已有negative采集，不另行重采')
    parser.add_argument('--inverse-probe',action='store_true',help='在持卡锁的边界分别保存求逆原版的inverse、info及CPU/NPU观察')
    parser.add_argument('--scan-stability',action='store_true',help='使用冻结反例重放扫描参考的默认/确定性重复稳定性，不修改正式合同')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+',args.negative_proposal):parser.error('候选名称必须为单个目录名')
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=Runner(args.batch)
        try:
            r.wait_idle()
            marker=r.q.runtime/'STOP-AFTER-JOB'
            if marker.exists():
                value=load(marker)
                if value.get('handoff_token')!=args.handoff_token:raise r.module.Attention('交接标记不属于本次控制器，不自动清除')
                archived=marker.with_name('handoff-'+args.handoff_token+'-'+str(time.time_ns())+'.json');marker.rename(archived)
                r.state.setdefault('handoffs',[]).append(dict(marker=str(archived),time=r.module.now()))
            r.state['status']='running';r.save()
            op='negative_'
            from validation_status import supplement_status
            # 新增NPU正负零检查后，不让旧的小测试集继续为最佳实现背书。
            old=r.state['operators'].get(op,{}).get('best') or {}
            if old:
                gates=Path(old['extra_checks']) if old.get('extra_checks') else None
                qualified=supplement_status(op,load(gates),old['source_sha256']) if gates and gates.exists() else 'not_run'
                if qualified!='passed':
                    fresh=r.batch/'results'/op/('requalification-'+old['source_sha256'][:12])/'target-extra-checks-v2.json'
                    fresh.parent.mkdir(parents=True,exist_ok=True)
                    if not fresh.exists():
                        r.job('requalify:'+op+':'+old['source_sha256'][:12],lambda:r.q.run('requalify-negative',
                            quote([PY,HERE/'validate.py','--operator',op,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
                                   '--source',old['source'],'--output',fresh]),1200))
                    qualified=supplement_status(op,load(fresh),old['source_sha256']) if fresh.exists() else 'not_run'
                    if qualified=='passed':old['extra_checks']=str(fresh)
                    else:
                        rejected=dict(old,qualification=qualified,requalification=str(fresh))
                        r.state['operators'][op].setdefault('retained_unqualified',[]).append(rejected)
                        r.state['operators'][op].pop('best',None)
                    r.save()
            proposal=r.batch/'proposals'/op/args.negative_proposal/'main.py'
            suffix=args.negative_proposal.removeprefix('launch-reuse-')
            label='HG-KDA-bitwise' if suffix=='bitwise-v2' else 'HG-KDA-'+suffix
            if proposal.exists():
                result=r.job('proposal:negative_:'+suffix,lambda:r.evaluate(op,proposal,proposal.parent/'formal',label))
                if result and result.get('status')=='PASSED':
                    gates=proposal.parent/'extra-checks.json'
                    if not gates.exists():
                        r.job('proposal-extra:negative_:'+suffix,lambda:r.q.run('proposal-negative-'+suffix+'-extra',
                            quote([PY,HERE/'validate.py','--operator',op,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
                                   '--source',proposal,'--output',gates]),1200))
                    if gates.exists() and supplement_status(op,load(gates),result['source_sha256'])=='passed':
                        result['extra_checks']=str(gates);r.consider_best(r.state['operators'][op],result);r.save()
                        from knowledge import organize
                        from policy import operator_type
                        formal=Path(result['result']);request=formal.with_name(formal.name.replace('.result.json','.request.json'))
                        record=dict(id=op+'-'+result['source_sha256'][:12],operator_type=operator_type(op),
                            action_id='validated-implementation-'+op,scope=dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'),
                            summary='复用历史启动方式并保留目标NPU正负零语义的实现；仅记录通过事实，不继承性能因果',
                            status='validated',updated_at=r.module.now(),evidence=dict(source=str(proposal),
                                source_sha256=result['source_sha256'],formal_result=str(formal),
                                formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),formal_request=str(request),
                                supplementary=[dict(path=str(gates),sha256=hashlib.sha256(gates.read_bytes()).hexdigest())]))
                        kp=r.batch/'knowledge/records.json';records=load(kp) if kp.exists() else []
                        if not any(entry.get('id')==record['id'] and entry.get('evidence')==record['evidence'] for entry in records):
                            records.append(record)
                        save_json(kp,records);save_json(kp.parent/'index.json',organize(records))
            # FP32、1024连续输入属于来源可见Ascend C参考的资格范围。
            row=r.state['operators'].get(op,{})
            best=row.get('best') or {}
            if best.get('status')=='PASSED' and not args.skip_negative_refresh:
                source=Path(best['source']);analysis=r.batch/'analysis'/op
                capture=analysis/('paired-validated-'+best['source_sha256'][:12]);spec=analysis/('spec-'+best['source_sha256'][:12]+'.json')
                if not spec.exists():
                    leaf=r.batch/r.q.records[op]['catalog_leaf']
                    cases=[__import__('json').loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
                    case=next(c for c in cases if c['inputs']['spec']['dtype']=='float32' and c['inputs']['spec']['shape']==[1024])
                    save_json(spec,dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        case=case,scope='经完整评测候选的代表输入；Ascend C自产参考仅作补充对照'))
                if not capture.exists():
                    r.job('paired:negative_:'+best['source_sha256'][:12],lambda:r.q.run('paired-negative-'+best['source_sha256'][:12],
                        quote([PY,HERE/'collect.py','--spec',spec,'--output',capture,'--simulator']),2400))
                context=analysis/('context-'+best['source_sha256'][:12]+'.json')
                if not context.exists() and (capture/'state.json').exists():
                    r.job('context:negative_:'+best['source_sha256'][:12],lambda:r.q.run('context-negative-'+best['source_sha256'][:12],
                        quote([PY,HERE/'assemble.py','--capture',capture,'--output',context]),600))
                if context.exists():
                    save_json(analysis/'latest-context.json',dict(path=str(context),sha256=hashlib.sha256(context.read_bytes()).hexdigest()))
            if args.scan_stability:
                historical=r.batch.parent/'two-closure-20261024/research/reference-stability/default-256x8-cumulative_trapezoid-input.bin'
                raw=historical.read_bytes()
                expected='f5874ea96d99dcc5b9399c8ccea5a50015d99d1e52936425b28d65e842b3ed19'
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('扫描参考反例原字节身份不符')
                output=r.batch/'analysis/cumulative_trapezoid/reference-stability-replay-v1'
                output.mkdir(parents=True,exist_ok=True)
                frozen=output/'input-fp16-256x8.bin'
                if frozen.exists() and frozen.read_bytes()!=raw:raise ValueError('扫描冻结输入已变化')
                if not frozen.exists():frozen.write_bytes(raw)
                script=TOOLS/'reproduce-cumsum-stability.py'
                provenance=output/'provenance.json'
                if not provenance.exists():
                    save_json(provenance,dict(input_sha256=expected,source=str(historical),
                        script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                        scope='重放历史冻结反例于本批固定环境；默认/官方确定性设置在不同进程单列'))
                for mode in ['default','deterministic']:
                    result=output/(mode+'.json')
                    args_mode=['--deterministic'] if mode=='deterministic' else []
                    if not result.exists():
                        r.job('scan-stability:'+mode,lambda mode=mode,result=result,args_mode=args_mode:r.q.run('scan-stability-'+mode,
                            quote([PY,script,'--input',frozen,'--shape','256','8','--kind','cumulative_trapezoid',
                                   '--repeats','12','--output',result]+args_mode),180))
            if args.inverse_probe:
                op='linalg_inv_ex'
                record=r.state['operators'].get(op,{}).get('baseline') or {}
                source=Path(record['source'])
                if hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
                    raise ValueError('求逆原版源码SHA不符')
                output=r.batch/'analysis'/op/('contract-probe-'+record['source_sha256'][:12]+'.json')
                if not output.exists():
                    r.job('inverse-contract-probe:'+record['source_sha256'][:12],lambda:r.q.run('inverse-contract-probe-original',
                        quote([PY,HERE/'probe_inverse.py','--source',source,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
                               '--output',output]),180))
            for op in args.host_profile:
                if op not in r.q.records:raise ValueError('未登记的宿主诊断目标 '+op)
                best=r.state['operators'].get(op,{}).get('best') or {}
                gates=Path(best['extra_checks']) if best.get('extra_checks') else None
                if not gates or not gates.exists() or supplement_status(op,load(gates),best.get('source_sha256'))!='passed':
                    continue
                output=r.batch/'analysis'/op/('host-profile-'+best['source_sha256'][:12]+'.json')
                if not output.exists():
                    r.job('host-profile:'+op+':'+best['source_sha256'][:12],lambda op=op,best=best,output=output:r.q.run('host-profile-'+op,
                        quote([PY,HERE/'host_profile.py','--source',best['source'],'--expected-sha',best['source_sha256'],
                               '--leaf',r.batch/r.q.records[op]['catalog_leaf'],'--output',output]),180))
            from capture_record import capture
            for key in args.paired_record:
                record=r.state.get('jobs',{}).get(key,{}).get('result')
                if not record:raise ValueError('补采记录不存在：'+key)
                capture(r,record,update_latest=False)
            for op in args.paired_best:
                if op not in r.q.records:raise ValueError('未登记补采目标：'+op)
                record=r.state.get('operators',{}).get(op,{}).get('best')
                if not record:raise ValueError('没有可补采的已验证最佳版本：'+op)
                capture(r,record,update_latest=True)
            for op in args.reference_audit:
                if op not in r.q.records:raise ValueError('未登记的参考审计目标：'+op)
                audit=r.job('reference-path:'+op,lambda op=op:r.audit_reference(op))
                row=r.state['operators'].setdefault(op,{})
                row['reference_path_status']=(audit or {}).get('status','unknown')
                best=row.get('best') or {}
                gates=Path(best['extra_checks']) if best.get('extra_checks') else None
                if (best.get('status')=='PASSED' and best.get('geo_mean',0)>.8 and gates and gates.exists()
                        and supplement_status(op,load(gates),best.get('source_sha256'))=='passed'):
                    row['status']='合同及补充检查通过，超过0.8×' if (audit or {}).get('eligible_default_comparison') else '数值和性能通过，参考路径需单列核验'
                r.save()
            r.execute()
        except Exception as error:
            r.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));r.save()
            print('RESUME_FAILED '+str(error),flush=True);raise


if __name__=='__main__':main()
