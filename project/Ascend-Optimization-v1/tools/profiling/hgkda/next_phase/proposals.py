"""在持卡锁的任务边界验证排队提案；提案数据不改变合同与评分。"""
import hashlib
import re
import time
from pathlib import Path

from validation_status import supplement_status


def extra_gate_path(out, spec):
    revision=spec.get('extra_check_revision')
    if revision is None:return out/'extra-checks.json',''
    if not isinstance(revision,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',revision):
        raise ValueError('非法补测版本；不得覆盖旧结果或使用路径分隔符')
    return out/('extra-checks-'+revision+'.json'),':'+revision


def process(r):
    from run import PY,HERE,quote,load,save_json
    for manifest in sorted((r.batch/'proposals').glob('*/*/proposal.json')):
        if time.time()>r.state.get('active_started_epoch',time.time())+3600*r.q.plan['duration_hours']:
            return
        spec=load(manifest);op=spec['operator'];name=spec['id']
        if op not in r.q.records:raise r.module.Attention('提案算子未登记 '+op)
        if not name.replace('-','').replace('_','').isalnum():raise ValueError('非法提案名称')
        source=(manifest.parent/spec['source']).resolve();source.relative_to(manifest.parent.resolve())
        if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_sha256']:
            raise r.module.Attention('排队提案源码改变 '+str(source))
        out=manifest.parent/'assessment'
        gates,gate_suffix=extra_gate_path(out,spec)
        best=r.state['operators'].get(op,{}).get('best') or {}
        if spec.get('only_if_best_ineligible') and best.get('extra_checks'):
            gates=Path(best['extra_checks'])
            if gates.exists() and supplement_status(op,load(gates),best['source_sha256'])=='passed':
                save_json(out/'not-needed.json',dict(status='not_needed',reason='当前最佳已通过补充合同，无需此语义补丁',source_sha256=best['source_sha256']))
                continue
        result=r.job('queued-proposal:'+op+':'+name,lambda:r.evaluate(op,source,out/'formal','HG-KDA-'+name))
        if not result or result.get('status')!='PASSED':continue
        if not gates.exists():
            r.job('queued-proposal-extra:'+op+':'+name+gate_suffix,lambda:r.q.run('proposal-'+op+'-'+name+'-extra',
                quote([PY,HERE/'validate.py','--operator',op,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
                       '--source',source,'--output',gates]),1200))
        if not gates.exists() or supplement_status(op,load(gates),result['source_sha256'])!='passed':continue
        result.update(extra_checks=str(gates),proposal=str(manifest),attribution=spec['attribution'])
        row=r.state['operators'].setdefault(op,{})
        parent=row.get('best') or {}
        if parent.get('extra_checks'):
            old_gates=Path(parent['extra_checks'])
            if not old_gates.exists() or supplement_status(op,load(old_gates),parent['source_sha256'])!='passed':
                row.setdefault('retained_unqualified',[]).append(dict(parent,qualification='supplement_failed_or_stale'))
                row.pop('best',None)
        r.consider_best(row,result)
        if (row.get('best') or {}).get('source_sha256')==result['source_sha256']:
            if result.get('geo_mean',0)<=.8:row['status']='完整合同及补测通过，性能未达标'
            elif row.get('reference_path_status')=='npu_paths_observed':row['status']='完整合同及补测通过，超过0.8×'
            else:row['status']='数值和性能通过，参考路径待核验'
        r.save()
        if (row.get('best') or {}).get('source_sha256')!=result['source_sha256']:
            # 完整结果保留；更慢的候选不替换当前成功实现知识。
            continue
        from knowledge import organize
        from policy import operator_type
        formal=Path(result['result']);request=formal.with_name(formal.name.replace('.result.json','.request.json'))
        record=dict(id=op+'-'+result['source_sha256'][:12],operator_type=operator_type(op),
            action_id='validated-implementation-'+op,scope=dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'),
            summary=spec.get('summary','通过本批完整合同与补测的提案实现'),status='validated',updated_at=r.module.now(),
            evidence=dict(source=str(source),source_sha256=result['source_sha256'],formal_result=str(formal),
                formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),formal_request=str(request),
                supplementary=[dict(path=str(gates),sha256=hashlib.sha256(gates.read_bytes()).hexdigest())]))
        path=r.batch/'knowledge/records.json';records=load(path) if path.exists() else []
        if not any(x.get('id')==record['id'] and x.get('evidence')==record['evidence'] for x in records):records.append(record)
        save_json(path,records);save_json(path.parent/'index.json',organize(records))
