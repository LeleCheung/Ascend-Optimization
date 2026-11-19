"""Attach evidence labels, keeping retrospective mechanism inference explicit."""
import hashlib
import json


def annotate(rows, root):
    path=root/'development/experiments/attribution-review-20261117/records.json'
    if not path.exists():
        return
    records=json.loads(path.read_bytes())['records']
    by_source={r['source_sha256']:r for r in records if r.get('source_sha256')}
    methods=json.loads((path.parent/'mainline-methods.json').read_bytes())
    confirmed_plans=json.loads((path.parent/'confirmed-plans.json').read_bytes())
    ledger_path=root/'development/reports/trace-attribution-matrix-20261116.json'
    ledger=json.loads(ledger_path.read_bytes())
    plans={}
    for item in ledger['rows']:
        for entry in item.get('historical_rounds',[]):
            if (entry.get('plan') or {}).get('kind')=='performance':
                plans.setdefault(entry.get('source_sha256'),[]).append(entry['plan'])
    for record in records:
        for ref in record.get('evidence',[]):
            assert hashlib.sha256((root/ref['path']).read_bytes()).hexdigest()==ref['sha256']
    for row in rows:
        mainline={}
        seen=set()
        for event in row.get('display_events',[]):
            sha=event['source_sha256'];loc=event.get('change_location',{})
            if event.get('is_baseline'):
                seen.add(sha);continue
            if sha in seen or loc.get('location')=='retest':
                event['attribution_status']='version_selection_or_retest'
                seen.add(sha);continue
            seen.add(sha)
            if sha[:8] not in methods:
                raise ValueError('Missing mainline method: '+row['operator']+' '+sha)
            method,primary=methods[sha[:8]]
            existing=by_source.get(sha)
            if existing and existing.get('role')=='guidance':
                record=dict(existing)
            else:
                linked=plans.get(sha,[])
                record=dict(operator=row['operator'],source_sha256=sha,role='guidance',method=method,
                            primary=primary,secondary=['ir'] if primary!='ir' else [],confidence='inferred',
                            location=loc.get('location','mixed'),evidence=[])
                if linked:
                    record['inference_basis']='recorded_plan'
                    record['reason']='优化计划明确记载改法；结合其中的假设与测量推断主要判断来源。'
                    record['plan_evidence']=linked
                    record['evidence'].append({'path':str(ledger_path.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest()})
                    if sha[:8] in confirmed_plans:
                        record['confidence']='confirmed'
                        record['reason']=confirmed_plans[sha[:8]]
                        if sha.startswith('1df2d942'):record['secondary']=['knowledge','ir']
                elif primary=='paired':
                    record['inference_basis']='historical_context'
                    record['reason']='根据历史方法标签、实验批次和源码变化推断 Profiling 参与判断；包含专项计时，不能据此断言已采集成对硬件计数器。'
                elif primary=='knowledge':
                    record['inference_basis']='historical_context'
                    record['reason']='根据相近实现的结构复用与历史迁移记录推断知识／经验复用；不等同于确认调用了知识库检索。'
                else:
                    record['inference_basis']='retrospective_code'
                    record['reason']='依据前后源码差异及版本说明推断修改机制，归为源码／IR分析；实际 Agent 当时的取证入口未完整记录，不表示已确认读取 IR 或排除其他工具。'
                source=root/loc['source_path']
                record['evidence'].append({'path':str(source.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
                record['changed_symbols']=loc.get('changed_symbols',{})
            record['event_id']=event['id']
            mainline[event['id']]=record
        row['attribution_review']=[r for r in records if r['operator']==row['operator']]
        row['mainline_attribution_review']=list(mainline.values())
        for key in ['display_events','full_history','events']:
            for event in row.get(key,[]):
                record=mainline.get(event['id']) or by_source.get(event['source_sha256'])
                if event.get('attribution_status')=='version_selection_or_retest':
                    event.pop('attribution',None)
                    continue
                if record and record.get('role')=='guidance':
                    event['attribution']=record
        if row.get('method_transfer'):
            record=by_source.get(row['method_transfer']['source_sha256'])
            if record:row['method_transfer']['attribution']=record


def tooltip(record):
    label={'confirmed':'日志确认','inferred':'推断'}[record['confidence']]
    names={'paired':'Profiling','sim':'模拟器','ir':'源码／IR','knowledge':'知识／经验'}
    return label+'：'+record['reason']+'；主来源：'+names[record['primary']]+'；辅助来源：'+('、'.join(names[x] for x in record.get('secondary',[])) or '无单独确认')
