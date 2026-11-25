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
                    record['reason']='The plan explicitly records the change; infer primary guidance from its hypotheses and measurements.'
                    record['plan_evidence']=linked
                    record['evidence'].append({'path':str(ledger_path.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest()})
                    if sha[:8] in confirmed_plans:
                        record['confidence']='confirmed'
                        record['reason']=confirmed_plans[sha[:8]]
                        if sha.startswith('1df2d942'):record['secondary']=['knowledge','ir']
                elif primary=='paired':
                    record['inference_basis']='historical_context'
                    record['reason']='Infer profiling participation from historical labels, batches, and source changes, including targeted timing; paired hardware counters are not thereby established.'
                elif primary=='knowledge':
                    record['inference_basis']='historical_context'
                    record['reason']='Infer knowledge/experience reuse from structural similarity and transfer records, without claiming confirmed knowledge retrieval.'
                else:
                    record['inference_basis']='retrospective_code'
                    record['reason']='Infer mechanism from source diffs/version notes as source/IR analysis. Actual tool inputs were incompletely recorded; IR reads or exclusion of other tools remain unresolved.'
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
    label={'confirmed':'Log-confirmed','inferred':'Inferred'}[record['confidence']]
    names={'paired':'Profiling','sim':'Simulator','ir':'Source / IR','knowledge':'Knowledge / experience'}
    return label+'：'+record['reason']+'; primary: '+names[record['primary']]+'; auxiliary: '+('、'.join(names[x] for x in record.get('secondary',[])) or 'None separately confirmed')
