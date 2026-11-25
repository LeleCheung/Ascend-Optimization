"""Reconstruct fixed-CANN-reference progress while retaining raw scores, invalid nodes, and exploratory branches."""
import argparse
import ast
import collections
import copy
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path

from .stitched import read, save, svg, method_label, export_check


def case_id(name):
    return '_'+name if name.startswith('conj_physical-') else name


def timing_contract(raw):
    result=[]
    for line in raw.decode('utf8').splitlines():
        if line.strip():
            item=json.loads(line);item['name']=case_id(item['name']);result.append(item)
    return {x['name']:x for x in result}


def gmean(values):
    values=list(values)
    if not values or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid timing')
    return math.exp(sum(math.log(v) for v in values)/len(values))


def rebase(anchor,result):
    """Use fixed per-case CANN timing to avoid mixing candidate repeat variation into aggregate score ratios."""
    def cases(d):
        rows=[x for x in d['per_workload'] if x.get('phase')=='timing']
        mapped={case_id(x['uuid']):x for x in rows}
        if len(mapped)!=len(rows):raise ValueError('duplicate timing case')
        return mapped
    a,b=cases(anchor),cases(result)
    if set(a)!=set(b):raise ValueError('timing case sets differ')
    rows=[]
    for name in sorted(a):
        if a[name].get('status')!='PASSED' or b[name].get('status')!='PASSED':raise ValueError('timing case failed')
        numerator=a[name].get('reference_latency_ms');denominator=b[name].get('latency_ms')
        if numerator is None or denominator is None:raise ValueError('missing case latency')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (numerator,denominator)):
            raise ValueError('nonpositive or nonfinite latency')
        ratio=numerator/denominator
        rows.append(dict(case=name,cann_anchor_ms=numerator,historical_candidate_ms=denominator,ratio=ratio))
    return dict(score=gmean(x['ratio'] for x in rows),cases=rows,
                formula='exp(mean(log(CANN_anchor_reference_ms / historical_candidate_ms)))',
                scope='Historical timings recalculated with fixed reference; not a new simultaneous CANN evaluation')


def normalize(row,root,linkdata,evidence,latencies,inventory):
    sys.path.insert(0,str(root/'hgkda/src'))
    from hgkda.gate import inspect_text
    op=row['operator'];current=root/'operators'/op
    anchor=read(current/'evidence/retained/evaluation/result.json')
    anchor_path='operators/'+op+'/evidence/retained/evaluation/result.json'
    contract=timing_contract((current/'contracts/timing.jsonl').read_bytes())
    settings=read(current/'contracts/relocation.json')['settings']
    final_bundle=read(current/'contracts/relocation.json')['original_bundle_id']
    linksroot=root/'development/experiments/lineage/20261117/links-v9'
    bundles={b['bundle_id']:b for b in linkdata['bundles']}
    originals=next(r for r in inventory['operators'] if r['operator']==op)
    docs=linkdata['documents'];negative={};approved={};sources={};scan={}
    current_checks={}
    for path in (root/'development/experiments/lineage/20261117/current-checks').glob('*.json'):
        check=read(path)
        if check.get('status')=='failed' and check.get('source_sha256'):
            current_checks[check['source_sha256']]=str(path.relative_to(root))
            negative.setdefault(check['source_sha256'],[]).append(str(path.relative_to(root)))
    manifest=read(current/'manifest.json')
    bad=manifest['versions'].get('historical_noncompliant',{}).get('source',{}).get('sha256')
    if bad:negative[bad]=['Historical version is known to violate strict Triton constraints']
    for doc in docs:
        o=doc['data']
        if not isinstance(o,dict):continue
        if Path(doc['path']).name == 'review-exclusions.json':
            for entry in o.get('records', []):
                if entry.get('source_sha256') and entry.get('status') == 'exclude_from_final_delivery':
                    negative.setdefault(entry['source_sha256'], []).append(doc['path'])
        sha=o.get('source_sha256')
        if not sha:continue
        status=str(o.get('status','')).lower()
        if status=='reviewed_strict_triton' and o.get('all_branches_reviewed') is True and o.get('dependencies_reviewed') is True:
            approved[sha]=doc['path']
        if status in ('failed','rejected','disqualified') or (isinstance(o.get('passed'),int) and isinstance(o.get('total'),int) and o['passed']<o['total']):
            negative.setdefault(sha,[]).append(doc['path'])
    for m in originals['measurements']:
        for s in m.get('sources',[]):sources[s['sha256']]=evidence/s['export']
    # Current delivery remains tied to its complete accepted evidence, not a heuristic gate.
    approved[row['delivery']['sha256']]=str(current/'manifest.json')
    anchor_check=rebase(anchor,anchor)
    assert abs(anchor_check['score']-row['delivery']['cann_score'])<1e-9
    conversions=[];excluded=[];candidates=[]
    for original in row['events']:
        e=copy.deepcopy(original);m=e['record'];sha=e['source_sha256']
        reason=None
        if e['is_baseline'] and sha in current_checks:
            e.update(raw_score=e['score'],raw_reference=e['reference'],score=None,
                     status='SUPPLEMENTARY_FAILED',reference='CANN',score_kind='failed_baseline',
                     additional_check=current_checks[sha])
            candidates.append(e);continue
        if e['is_baseline'] and e['score'] is None:
            e.update(raw_score=None,raw_reference=e['reference'],reference='CANN',score_kind='failed_baseline')
            candidates.append(e);continue
        if not e['is_baseline']:
            if e['status']!='PASSED':reason='formal_failure'
            elif m.get('baseline_path'):reason='repeated_baseline'
            elif '/kg-no-profile/' in m['result']['path'] or '/kg-profile/' in m['result']['path']:
                reason='independent_comparison_arm'
            elif any(s in e['stage'] for s in ('independent-skill','selu-skill-recovery','log10-method-transfer','overnight-regions')):
                reason='independent_regeneration_branch'
            elif sha in negative and sha!=row['delivery']['sha256']:reason='failed_check_or_known_noncompliant'
            elif m.get('measurement',{}).get('is_hack') is True:reason='admission_rejected'
        if not reason and m.get('ledger_round'):reason='ledger_without_per_case_times'
        if reason:
            excluded.append(dict(event=e,reason=reason,evidence=negative.get(sha,[])));continue
        if not e['is_baseline'] and sha not in approved:
            if sha not in sources:reason='source_unavailable_for_screen'
            else:
                if sha not in scan:
                    raw=sources[sha].read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
                    scan[sha]=inspect_text(raw.decode('utf8'))
                if scan[sha]['issues'] or not scan[sha]['triton_kernels']:reason='path_review_pending'
        if reason:
            excluded.append(dict(event=e,reason=reason,static_screen=scan.get(sha)));continue
        if not e['is_delivery']:
            binding=(m.get('contract',{}).get('binding') or {}).get('bundle_id')
            identity=None
            if binding==final_bundle:identity=dict(kind='current_contract_bundle',bundle_id=binding)
            elif binding in bundles:
                bundle=bundles[binding]
                timing=next((x for x in bundle['members'] if x['member'].endswith('timing.jsonl')),None)
                if timing and timing_contract((linksroot/timing['export']).read_bytes())==contract:
                    identity=dict(kind='equal_timing_spec_and_seed',bundle_id=binding,timing_sha256=timing['sha256'])
            if identity is None:
                excluded.append(dict(event=e,reason='timing_contract_not_matched'));continue
            measured_settings=m.get('contract',{}).get('settings') or {}
            if any(measured_settings.get(k)!=settings.get(k) for k in ('warmup_ms','benchmark_ms','num_trials')):
                excluded.append(dict(event=e,reason='measurement_settings_changed'));continue
            f=latencies/(m['result']['sha256']+'.json')
            if not f.exists():
                excluded.append(dict(event=e,reason='missing_per_case_result'));continue
            raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==m['result']['sha256']
            result=json.loads(raw)
        else:result=anchor;identity=dict(kind='retained_delivery')
        try:converted=rebase(anchor,result)
        except (ValueError,ZeroDivisionError) as err:
            excluded.append(dict(event=e,reason='incomplete_case_timings',error=str(err)));continue
        e.update(raw_score=e['score'],raw_reference=e['reference'],score=converted['score'],reference='CANN',
                 score_kind='measured' if e['is_delivery'] else 'rebased',conversion=converted,
                 timing_identity=identity,source_admission='reviewed_strict' if sha in approved else ('native_baseline' if e['is_baseline'] else 'static_screen_passed_not_full_review'))
        conversions.append(e);candidates.append(e)
    baseline=next((e for e in candidates if e['is_baseline']),None)
    if baseline is None:
        # Do not disguise a missing conversion as an original correctness failure.
        baseline=copy.deepcopy(next(e for e in row['events'] if e['is_baseline']))
        baseline.update(raw_score=baseline['score'],raw_reference=baseline['reference'],score=None,
                        score_kind='unconverted_baseline',reference='CANN')
    final=next(e for e in candidates if e['is_delivery'])
    # Select fixed first measurements, never the fastest repeated timing of a source.
    bysha={}
    for e in sorted(candidates,key=lambda e:e['sort']):
        if not e['is_baseline'] and not e['is_delivery']:bysha.setdefault(e['source_sha256'],e)
    displayed=[baseline];seen={baseline['source_sha256']};moved=[]
    for old in row['display_events']:
        sha=old['source_sha256']
        if old['is_baseline'] or old['is_delivery'] or sha in seen or sha==final['source_sha256']:continue
        e=bysha.get(sha)
        if e is None:continue
        seen.add(sha);displayed.append(e)
    displayed.append(final)
    # This is a presentation summary of useful milestones, not a rewritten search log.
    # Summarize improving milestones, explicitly retain omitted regressions in details;
    # keep the final measured endpoint even when it regresses.
    retained=[displayed[0]]
    for i,e in enumerate(displayed[1:-1],1):
        previous=retained[-1]
        if previous['score'] is not None and e['score']<previous['score']:
            moved.append(dict(event=e,reason='valid_exploration_regression'));continue
        retained.append(e)
    retained.append(final)
    displayed=retained
    # Shared final source appears once, at its accepted final measurement.
    adjacent=collections.defaultdict(list)
    for edge in row['edges']:adjacent[edge['parent']].append(edge)
    def connect(a,b):
        if a['source_key']==b['source_key']:return dict(kind='same_source',proof=[])
        q=collections.deque([(a['source_key'],[])]);visited={a['source_key']}
        while q:
            node,path=q.popleft()
            for edge in adjacent[node]:
                if edge['child']==b['source_key']:return dict(kind='recorded_lineage',proof=path+[edge])
                if edge['child'] not in visited:visited.add(edge['child']);q.append((edge['child'],path+[edge]))
        return dict(kind='unresolved',proof=None)
    connections=[dict(start=a['id'],end=b['id'],reference_change=False,**connect(a,b)) for a,b in zip(displayed,displayed[1:])]
    result=copy.deepcopy(row)
    result.update(display_events=displayed,connections=connections,normalized_view=True,
                  conversions=conversions,excluded=excluded,moved_exploration=moved,static_screen=scan,
                  anchor=dict(path=anchor_path,sha256=hashlib.sha256((root/anchor_path).read_bytes()).hexdigest(),
                              settings=settings,timing_spec=contract,score=row['delivery']['cann_score']),
                  view_policy="Qualified delivery-progress summary with explicit conversions, selection, and branches; not the controller's raw round trace")
    return extend_recovery(result,root,anchor)


def extend_recovery(row,root,anchor):
    """Append explicitly selected, fully checked recoveries without rewriting delivery."""
    base=root/'development/experiments/lineage/20261117'
    transfer_file=base/'method-transfers.json'
    transfer=read(transfer_file).get(row['operator']) if transfer_file.exists() else None
    if transfer:
        path=base/'links-v9/artifacts'/(transfer['document']+'.json')
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==transfer['document']
        doc=json.loads(raw);parent=doc.get('parent_source_sha256') or doc.get('parent_sha256')
        row['method_transfer']=dict(**transfer,parent_sha256=parent,source_sha256=doc['source_sha256'],
            evidence=str(path.relative_to(root)),scope='Explicit donor-source inheritance; does not establish the previous same-name operator as parent')
        target=doc['source_sha256']
        if transfer.get('continuation_document'):
            continuation=base/'links-v9/artifacts'/(transfer['continuation_document']+'.json')
            raw=continuation.read_bytes();assert hashlib.sha256(raw).hexdigest()==transfer['continuation_document']
            next_doc=json.loads(raw);assert next_doc['parent_source_sha256']==target
            target=next_doc['source_sha256']
            row['method_transfer']['continuation_evidence']=str(continuation.relative_to(root))
        exact=next(e for e in row['events'] if e['source_sha256']==target)
        first=next(e for e in row['display_events'] if e['source_key']==exact['source_key'])
        row['method_transfer']['display_target_sha256']=first['source_sha256']
        row['method_transfer']['display_targets']=[first['source_sha256']]
        if transfer.get('earlier_donor_sha256'):
            donor_sha=transfer['earlier_donor_sha256'];donor=base/'links-v9/artifacts'/(donor_sha+'.py')
            raw=donor.read_bytes();assert hashlib.sha256(raw).hexdigest()==donor_sha
            donor_key='ast:'+hashlib.sha256(ast.dump(ast.parse(raw)).encode()).hexdigest()
            earlier=next(e for e in row['display_events'] if e['source_sha256']==transfer['earlier_target_sha256'])
            queue=collections.deque([(donor_key,[])]);seen={donor_key};proof=None
            while queue:
                current,chain=queue.popleft()
                if current==earlier['source_key']:proof=chain;break
                for edge in row['edges']:
                    if edge['parent']==current and edge['child'] not in seen:
                        seen.add(edge['child']);queue.append((edge['child'],chain+[edge]))
            assert proof,'earlier donor branch must be supported by recorded source edges'
            row['method_transfer']['earlier_branch_proof']=proof
            row['method_transfer']['display_targets'].insert(0,earlier['source_sha256'])
        targets=set(row['method_transfer']['display_targets'])
        events={e['id']:e for e in row['display_events']}
        for connection in row['connections']:
            if connection['kind']=='unresolved' and events[connection['end']]['source_sha256'] in targets:
                connection.update(kind='documented_method_switch',source_inheritance=False,
                                  proof=row['method_transfer'],
                                  explanation='Method switch in performance history; code parent is the donor, not the previous same-name version')
    selection_file=base/'recovery-selections.json'
    choice=read(selection_file).get(row['operator']) if selection_file.exists() else None
    if not choice:return row
    folder=base/'recovery-results-v4'/choice['run']/('formal-'+choice['source_prefix'])
    audit=read(folder/'audit-final.json')['rows'][0]
    assert audit['full_passed'] and all(c['status']=='passed' for c in audit['checks'].values())
    source=folder/'candidate.py';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==audit['source_sha256']
    request=read(folder/'request.json')
    from hgkda.adapters.bundle import pack
    _,bundle=pack(root/'operators'/row['operator']/'contracts',exclude={'relocation.json'})
    assert request['binding']['bundle_id']=='sha256:'+bundle
    assert all(request['settings'][k]==row['anchor']['settings'][k] for k in ('warmup_ms','benchmark_ms','num_trials'))
    raw=read(folder/'evaluation/result.json');converted=rebase(anchor,raw)
    final=row['display_events'][-1];proof=[]
    if choice['kind']=='historical_restore':
        assert any(e['source_sha256']==sha for e in row['events'])
        proof=[dict(kind='restored_existing_source',source_sha256=sha,
                    audit=str((folder/'audit-final.json').relative_to(root)),scope=choice['scope'])]
    else:
        parent=final['source_sha256']
        for rel in choice['provenance']:
            path=base/rel;identity=read(path)
            assert identity['input_sha256']==parent
            parent=identity['output_sha256']
            proof.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**identity))
        assert parent==sha
    event=dict(id='recovery-'+sha[:16],source_sha256=sha,source_key=sha,score=converted['score'],
               raw_score=audit['geo_mean'],raw_reference='CANN',reference='CANN',score_kind='rebased',
               is_baseline=False,is_delivery=False,is_recovery=True,status='PASSED',
               stage='midday-recovery-20261117',conversion=converted,
               recovery_method={k:choice[k] for k in ('method','color','location')},
               record=dict(result=dict(path=str(folder/'evaluation/result.json'))))
    row['display_events'].append(event)
    row['connections'].append(dict(start=final['id'],end=event['id'],kind='recorded_recovery',
                                   reference_change=False,proof=proof,recovery_kind=choice['kind']))
    row['recovery']=dict(selection=choice,source_sha256=sha,original_delivery_retained=True,
                         evaluation=str((folder/'evaluation/result.json').relative_to(root)))
    return row


REASONS={
    'formal_failure':'Formal evaluation failed',
    'repeated_baseline':'Repeated baseline, not later optimization',
    'independent_regeneration_branch':'Independent regeneration branch',
    'independent_comparison_arm':'Independent control arm; separate from this chain',
    'failed_check_or_known_noncompliant':'Supplementary failure / known noncompliance',
    'admission_rejected':'Submission review rejected',
    'path_review_pending':'Path review pending; noncompliance not established',
    'ledger_without_per_case_times':'Ledger lacks per-case timings; retain raw record',
    'timing_contract_not_matched':'Performance-case identity unverified',
    'measurement_settings_changed':'Timing settings differ',
    'source_unavailable_for_screen':'Submitted source not located',
    'missing_per_case_result':'Per-case results not located',
    'incomplete_case_timings':'Per-case timing incomplete',
    'valid_exploration_regression':'Qualified exploratory regression moved to branch',
}


def render(rows,out,root):
    catalog=read(root/'development/reports/operator-traces-26-20261116/catalog.json');known={}
    for r in catalog['operators']:
        for a in r['view']['annotations']:
            p=r['points'][a['point']-1]
            loc=next((x['location'] for x in r['view'].get('nodes',[]) if x['point']==a['point']),'mixed')
            known[p.get('source_sha256')]=dict(method=a['method'],color=a['color'],location=loc)
    css="body{margin:28px;background:#fff7e8;color:#1b2434;font:19px/1.65 'Microsoft YaHei',sans-serif}h1{font-size:34px;overflow-wrap:anywhere}h2{font-size:27px;line-height:1.35;margin:8px 0 20px;overflow-wrap:anywhere}a{color:#276980;text-decoration:none;overflow-wrap:anywhere;transition:color .15s,background .15s}a:hover{color:#164559;text-decoration:underline;text-underline-offset:5px}a:focus-visible{outline:3px solid #276980;outline-offset:4px;border-radius:4px}body>p>a{display:inline-block;padding:5px 12px;margin:3px 0;background:#eee7d9;border-radius:7px;font-weight:600}body>p>a:hover{background:#e2d8c6;text-decoration:none}.viewport{overflow:auto;border:1px solid #ded7c9}.viewport>svg{display:block;max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:17px}td,th{border-bottom:1px solid #d8d5c8;padding:10px;text-align:left;vertical-align:top}details{margin:20px 0}summary{cursor:pointer;font-weight:700}.cards{display:grid;grid-template-columns:minmax(0,1fr);gap:24px}.cards img{width:100%}.card{min-width:0;border:1px solid #ded7c9;border-radius:12px;padding:20px;background:#fffaf0}.card h2 a{display:block;color:#253e4c}.card h2 a:hover{color:#276980}.cards img{display:block}code{overflow-wrap:anywhere}@media(min-width:1200px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:640px){body{margin:14px}.card{padding:12px}h1{font-size:28px}h2{font-size:23px}}"
    cards=[];notes=['# Qualified progress under a common CANN reference','','Updated 2026-11-17. The main view summarizes qualified progress, retaining original events, failures, and regressions. Failed native starts may use crosses; later invalid/pending candidates stay in details.',
        '', '## Conversion','',
        '`recalculated_speedup = exp(mean(log(delivery_per_case_CANN_time / historical_per_case_candidate_time)))`.',
        '', 'Fix the delivery CANN timings. Verify case configuration, seed, set, warmup, timing duration, and trials before recalculation. Do not divide aggregate scores or fill missing values with zero. _conj_physical normalizes name prefixes only.',
        '', '**Approximate markers denote recalculation; delivery uses its original measured CANN score.** Fixed references remove denominator switches, not historical load/clock/environment/noise. No new hardware experiment or temporal causality is established. Preserve original accuracy/timeout settings separately from timing compatibility.',
        '', '## Selection','',
        '- Move failures/known noncompliance to details. Pending full path review also stays outside the main view without being labeled noncompliant.',
        '- No-profiling/profiling controls, repeated baselines, and independent regeneration are distinct chains.',
        '- Qualified non-improving exploration moves to explicitly labeled branches. This retrospective view does not impersonate original controller decisions.',
        '- Always retain the fixed delivery endpoint, even below a historical recalculation. Preserve scores and native baselines without enforcing monotonicity.',
        '- Preserve ancestry evidence; conversion/selection does not fill missing parents.',
        '', '| Operator | Main-view nodes | Start | Measured delivery CANN |','| --- | ---: | --- | ---: |']
    for row in rows:
        op=row['operator'];folder=out/op
        for name in ['index.html','mainline.svg','mainline.png']:
            original=folder/name;backup=folder/('raw-'+name)
            if original.exists() and not backup.exists():backup.write_bytes(original.read_bytes())
        chart=svg(row,known)
        attribution_html=''
        attribution_records=row.get('mainline_attribution_review',[])+[r for r in row.get('attribution_review',[]) if r['source_sha256'] not in {a['source_sha256'] for a in row.get('mainline_attribution_review',[])}]
        if attribution_records:
            from .attribution import tooltip
            save(folder/'method-attribution.json',attribution_records)
            entries=[]
            for record in attribution_records:
                title=record['method']+('*' if record['confidence']=='inferred' else '')
                entries.append('<tr><td>'+html.escape(title)+'</td><td>'+html.escape(tooltip(record))+'</td><td>'+html.escape(record.get('outcome',record.get('scope','')))+'</td></tr>')
            attribution_html='<details><summary>Per-node method sources and inference</summary><p>* denotes inference; distinguish historical context from retrospective source diffs. Unstarred entries have explicit records. Diffs reveal mechanisms but do not prove which tool the agent read.</p><table><tr><th>Method</th><th>Evidence</th><th>Result / scope</th></tr>'+''.join(entries)+'</table><p><a href="method-attribution.json">Attribution records</a></p></details>'
            if any(r['confidence']=='inferred' for r in row['attribution_review']):
                (folder/'method-history.svg').write_text(svg(row,known,True),encoding='utf-8')
                attribution_html=attribution_html.replace('</details>','<p><a href="method-history.svg">Historical view with inference markers</a></p></details>')
        (folder/'mainline.svg').write_text(chart,encoding='utf8');save(folder/'cann-view.json',row)
        main=[]
        for i,e in enumerate(row['display_events']):
            score=('≈' if e.get('score_kind')=='rebased' else '')+f"{e['score']:.6f}×" if e['score'] is not None else 'Native failed'
            original_score=f"{e['raw_score']:.6f}× {e['raw_reference']}" if e.get('raw_score') is not None else str(e['status'])
            check_link=''
            if e.get('additional_check'):
                raw=(root/e['additional_check']).read_bytes();name='additional-check-'+hashlib.sha256(raw).hexdigest()[:12]+'.json'
                (folder/name).write_bytes(raw)
                check_link=' · <a href="'+name+'">Failed supplementary checks: evidence</a>'
            location_names={'kernel':'Device kernel','host':'Host','mixed':'Both','baseline':'Native start','retest':'Same-implementation repeat'}
            location=location_names[e['change_location']['location']]
            main.append('<tr><td>'+str(i)+'</td><td>'+score+'</td><td>'+html.escape(original_score)+check_link+'</td><td><code>'+e['source_sha256'][:12]+'</code></td><td>'+html.escape(e['stage'])+'</td><td>'+location+'</td></tr>')
        details=[]
        for x in row['excluded']+row['moved_exploration']:
            e=x['event'];value=e.get('raw_score',e.get('score'));ref=e.get('raw_reference',e['reference'])
            details.append('<tr><td><code>'+e['source_sha256'][:12]+'</code></td><td>'+html.escape(REASONS.get(x['reason'],x['reason']))+'</td><td>'+('No valid score' if value is None else f'{value:.6f}× '+ref)+'</td><td>'+html.escape(e['stage'])+'</td></tr>')
        transfer_html=''
        if row.get('method_transfer'):
            transfer=row['method_transfer']
            (folder/'method-transfer.json').write_bytes((root/transfer['evidence']).read_bytes())
            transfer_html=(f'<h2>Implementation source changed here</h2><p><a href="../{transfer["donor"]}/index.html">{transfer["donor"]}</a>'
                           f' → {html.escape(transfer["label"])} → {op}</p>'
                           '<p>The donor relation is recorded explicitly; this is not a direct edit of the previous same-name version. Preserve performance history and follow the actual donor for ancestry.'
                           '<a href="method-transfer.json">Original transfer record</a></p>')
        page=(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{op} qualified progress</title><style>{css}</style>'
              '<p><a href="../index.html">26-operator overview</a> | <a href="mainline.png">PNG</a> | <a href="mainline.svg">SVG</a> | <a href="cann-view.json">Conversion/selection evidence</a> | <a href="../recovery.html">Current experiments</a></p>'
              '<div class="viewport">'+chart+'</div>'+transfer_html+attribution_html+
              '<p>Approximate markers denote fixed-CANN recalculation; delivery uses original acceptance. Current research points have not replaced delivery artifacts. Hidden exploration remains available.</p>'
              '<details><summary>Main-view values: recalculated and raw scores</summary><p>Node shape classifies adjacent displayed source changes; gray marks native/repeat implementations. See change_location for functions and source SHA.</p><table><tr><th>Node</th><th>CANN scale</th><th>Raw score</th><th>Source</th><th>Batch</th><th>Change location</th></tr>'+''.join(main)+'</table></details>'
              '<details><summary>Historical details, including pending review and qualified regressions</summary><table><tr><th>Source</th><th>Reason</th><th>Raw score</th><th>Batch</th></tr>'+''.join(details)+'</table></details>'
              '<details><summary>Raw exploration and all evidence</summary><p><a href="raw-mainline.svg">Previous raw mainline</a> | <a href="history.svg">Raw history</a> | <a href="trace.json">Raw trace data</a></p></details></html>')
        (folder/'index.html').write_text(page,encoding='utf8')
        cards.append(f'<div class="card"><a href="{op}/index.html" aria-label="{op} trace details">'+chart.replace('<svg ','<svg style="display:block;width:100%;height:auto" ',1)+'</a></div>')
        first=row['display_events'][0];label='Native failed' if first['score'] is None else f"≈{first['score']:.3f}×"
        notes.append(f'| [{op}]({op}/index.html) | {len(row["display_events"])} | {label} | {row["delivery"]["cann_score"]:.3f}× |')
    index=out/'index.html'
    if index.exists() and not (out/'raw-index.html').exists():(out/'raw-index.html').write_bytes(index.read_bytes())
    recovery_link=' | <a href="recovery.html">Current recovery/optimization</a>' if (out/'recovery.html').exists() else ''
    recovery_link+=' | <a href="attribution-share.svg">Method-source share</a>'
    index.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>26-operator qualified progress</title><style>'+css+'</style><h1>26-operator qualified progress | common CANN reference</h1><p>Approximate historical recalculations; measured delivery endpoints. Only native starts may use failure crosses.</p><p><a href="cann-view-method.md">Conversion/selection method</a> | <a href="cann-view.json">Structured evidence</a>'+recovery_link+'</p><div class="cards">'+''.join(cards)+'</div></html>',encoding='utf8')
    (out/'cann-view-method.md').write_text('\n'.join(notes)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--png',action='store_true')
    a=p.parse_args();root=Path.cwd();out=a.output.resolve()
    evidence=root/'development/experiments/lineage/20261117/source-evidence'
    latencies=root/'development/experiments/lineage/20261117/latency-results'
    links=read(root/'development/experiments/lineage/20261117/links-v9/links.json')
    inventory=read(root/'development/reports/native-to-delivery-26-20261117/inventory.json')
    graph=read(out/'graph.json')
    rows=[normalize(r,root,links,evidence,latencies,inventory) for r in graph['operators']]
    from .change_locations import annotate
    annotate(rows,root)
    from .attribution import annotate as annotate_attribution
    annotate_attribution(rows,root)
    save(out/'cann-view.json',dict(operators=rows))
    render(rows,out,root)
    from .progress_view import update_catalog
    if out==root/'development/reports/operator-traces-stitched-20261117':
        update_catalog(root,png=a.png)
    elif a.png:export_check(out)
    print(json.dumps({'operators':len(rows),'mainline_nodes':sum(len(r['display_events']) for r in rows),
        'rebased_points':sum(len(r['conversions']) for r in rows),'excluded':dict(collections.Counter(x['reason'] for r in rows for x in r['excluded'])),
        'exploratory_regressions_moved':sum(len(r['moved_exploration']) for r in rows)},ensure_ascii=False))


if __name__=='__main__':main()
