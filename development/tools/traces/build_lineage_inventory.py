"""Index original FlagGems submissions, historical versions, and deliveries without guessing missing ancestry."""
import argparse
import ast
import collections
import hashlib
import html
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


EARLY = {'selu_', 'rad2deg_', 'log10_', 'uniform', 'value_selecting_reduction_backward',
         'scatter_add', 'upsample_bilinear2d', '_conj_physical'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path.cwd(); evidence=args.evidence.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    discovered=read(evidence/'inventory.json')
    selected=[x for x in read(root/'operators/index.json')['rows'] if x['scope']=='main_30' and x['accepted']]
    matrix={r['operator']:r for r in read(root/'development/reports/trace-attribution-matrix-20261116.json')['rows']}
    rows=[]

    def local_link(path,label):
        return '['+label+']('+Path(os.path.relpath(path,out)).as_posix()+')'

    def verify_export(record):
        if record.get('export'):
            assert sha((evidence/record['export']).read_bytes())==record['sha256'],record

    def baseline_identity(record, upstream):
        if not record.get('sources'):return dict(status='missing_submitted_source')
        source=record['sources'][0];verify_export(source)
        actual=ast.parse((evidence/source['export']).read_text(encoding='utf-8-sig')).body
        original=ast.parse(upstream.read_text(encoding='utf-8-sig')).body
        unchanged=len(actual)>=len(original) and all(ast.dump(a)==ast.dump(b) for a,b in zip(original,actual))
        return dict(status='original_module_plus_adapter' if unchanged else 'requires_review',
                    original_module_ast_unchanged=unchanged,
                    appended_statements=[ast.unparse(x) for x in actual[len(original):]] if unchanged else [],
                    scope='Module body is statically unchanged; dynamic imports/backend dispatch still need verification')

    for selection in selected:
        op=selection['operator']
        records=[r for r in discovered['records'] if r['operator']==op]
        campaign='night-20261021-v3' if op in EARLY else 'next25-20261029-v2'
        plan=next(x for x in discovered['plans'] if x['operator']==op and x['campaign']==campaign)
        upstream=root/'development/dependencies'/plan['record']['upstream_source'].replace('\\','/')
        assert sha(upstream.read_bytes())==plan['record']['upstream_source_sha256']
        baseline_records=[r for r in records if r['baseline_path']]
        canonical=[r for r in baseline_records if '/'+campaign+'/' in r['result']['path'] and
                   ('/FlagGems/' in r['result']['path'] if op in EARLY else ('/baseline/' if op=='negative_' else '/baseline-fixed/') in r['result']['path'])]
        assert len(canonical)==1,(op,len(canonical))
        baseline=canonical[0]
        identity=baseline_identity(baseline,upstream)
        assert identity['original_module_ast_unchanged'],op
        for r in records:
            verify_export(r['result'])
            for s in r.get('sources',[]):verify_export(s)
            for s in r.get('provenance',[]):verify_export(s)
            if r.get('request'):verify_export(r['request'])
        baseline_raw=read(evidence/baseline['result']['export'])
        baseline['result_log_excerpt']=str(baseline_raw.get('log',''))[-2200:]
        baseline['identity']=identity
        for r in baseline_records:
            r['identity']=baseline_identity(r,upstream)
        manifest=read(root/'operators'/op/'manifest.json')
        delivery=dict(manifest['selected'])
        assert sha((root/delivery['path']).read_bytes())==delivery['sha256']
        result_path=root/'operators'/op/'evidence/retained/evaluation/result.json'
        result=read(result_path)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']
        assert abs(result['geo_mean']-delivery['cann_score'])<1e-12
        delivery.update(result=result_path.relative_to(root).as_posix(),result_sha256=sha(result_path.read_bytes()),
                        passed=result['num_passed'],total=result['num_workloads'])
        matches=[r for r in records if any(s['sha256']==delivery['sha256'] for s in r.get('sources',[]))]
        assert matches,op
        exact=[r for r in matches if r['measurement']['geo_mean']==delivery['cann_score']]
        delivery['remote_exact_measurements']=[r['result'] for r in exact]
        delivery['remote_same_source_measurements']=[r['result'] for r in matches]
        noncompliant=manifest['versions'].get('historical_noncompliant',{}).get('source',{}).get('sha256')
        for r in records:
            names=[s['sha256'] for s in r.get('sources',[])]
            path=r['result']['path'];impl=r.get('implementation',{}).get('name') or ''
            r['role']='baseline' if r['baseline_path'] else ('reference' if '/references/' in path or '/references-v' in path or impl.startswith('CANN-') else 'candidate')
            r['constraint_status']='known_noncompliant' if noncompliant and noncompliant in names else ('selected_strict_delivery' if delivery['sha256'] in names else 'historical_not_reaudited')
        candidate_records=[r for r in records if r['role']=='candidate']
        historical=matrix[op]['historical_rounds']
        source_refs=collections.defaultdict(list)
        for r in records:
            for s in r.get('sources',[]):source_refs[s['sha256']].append(s['export'])
        ledger_entries=[]
        for h in historical:
            item=dict(h)
            item['recovered_sources']=sorted(set(source_refs.get(h.get('source_sha256'),[])))
            ledger_entries.append(item)
        # Reference bundle IDs are compared as recorded; relocation can also change IDs.
        base_bundle=baseline.get('contract',{}).get('binding') or {}
        same_bundle=[r for r in exact if (r.get('contract',{}).get('binding') or {})==base_bundle]
        gap=[]
        if baseline['measurement']['status']!='PASSED':gap.append('Native start did not fully pass; preserve failure without assigning zero')
        if not same_bundle:gap.append('Native/delivery reference or contract archives differ; separate segments until common-scope verification or measurement')
        gap.append('Source and measurement located; cross-batch parents and method sources still need segment review')
        row=dict(operator=op,upstream=dict(commit='d6a8eec473517a3d68157b208eb9c057eb1d4c50',
                    path=upstream.relative_to(root).as_posix(),sha256=sha(upstream.read_bytes()),plan=plan),
                 baseline=baseline,baseline_alternatives=baseline_records,delivery=delivery,
                 historical_ledger_entries=ledger_entries,measurements=records,
                 counts=dict(measurements=len(records),candidate_measurements=len(candidate_records),
                     candidate_sources=len(set(s['sha256'] for r in candidate_records for s in r.get('sources',[]))),
                     ledger_points=len(historical),baseline_records=len(baseline_records)),gaps=gap)
        rows.append(row)

    summary=dict(operators=len(rows),original_sources_verified=len(rows),baseline_records=sum(r['counts']['baseline_records'] for r in rows),
        canonical_baseline_passed=sum(r['baseline']['measurement']['status']=='PASSED' for r in rows),
        delivery_source_and_result_verified=len(rows),measurements=sum(r['counts']['measurements'] for r in rows),
        retrieval_errors=discovered['errors'])
    dump(out/'inventory.json',dict(schema='hgkda.native-to-delivery-inventory/v1',summary=summary,operators=rows))
    head=['# 26 operators: native start, historical versions, delivery','','Date: 2026-11-17. Read-only archive review without new device experiments.',
          '', 'Pinned FlagGems: `d6a8eec473517a3d68157b208eb9c057eb1d4c50`. All 26 original files match recorded SHA. Original module ASTs are unchanged in submitted baselines; only evaluation entries were appended. This does not verify dynamic backend dispatch.',
          '', '**Baseline submissions and delivered source/results were located for all 26 operators.** Native submissions differ from optimizer baselines that may already contain changes; older charts read only the latter.',
          '', 'Start scores use historical references while delivery scores use CANN; their ratio is not an optimization gain. Preserve failures. Measurement counts include batches/repeats, not model rounds; identify vendor references separately.',
          '', '| Operator | Native submission | Candidate measurements / sources | Delivery CANN | Full evaluation | Evidence |',
          '| --- | --- | ---: | ---: | ---: | --- |']
    pages=[]
    for row in rows:
        op=row['operator']; b=row['baseline']; d=row['delivery'];m=b['measurement'];c=row['counts']
        score=f"{m['geo_mean']:.3f}×" if m['status']=='PASSED' and m.get('geo_mean') is not None else m['status']
        head.append(f"| `{op}` | {score} | {c['candidate_measurements']}／{c['candidate_sources']} | {d['cann_score']:.3f}× | {d['passed']}/{d['total']} | [{op}]({op}.md) |")
        lines=['# '+op,'','## Original start','',
               '- '+local_link(root/row['upstream']['path'],'Pinned FlagGems source')+'；SHA `'+row['upstream']['sha256']+'`。',
               '- '+local_link(evidence/b['sources'][0]['export'],'Submitted source')+'; original module AST preserved, appended entry: `'+'; '.join(b['identity']['appended_statements']).replace('\n',' ')+'`。',
               '- '+local_link(evidence/b['result']['export'],'Original start result')+'：**'+score+'**；'+str(m.get('num_passed'))+'/'+str(m.get('num_workloads'))+'。',
               '- '+local_link(evidence/b['request']['export'],'Start request / contract binding')+'; server artifact: `'+b['result']['path']+'`。',
               '', '## Stages','',
               '| Stage | Available evidence |','| --- | --- |',
               '| Native submission | Original module plus evaluation entry; results above |',
               '| Early optimization | '+str(c['ledger_points'])+' ledger measurements; baseline may already include edits and cannot replace the native start |',
               '| Later changes / regressions | '+str(c['candidate_measurements'])+' candidate measurements, '+str(c['candidate_sources'])+' request sources; detailed locations below |',
               '| Delivery | '+f"{d['cann_score']:.6f}x CANN; source SHA and retained result verified, with matching remote source/score record |",
               '', '## All measurement entry points','',
               'List located measurements by storage, not inferred ancestry. inventory.json preserves remote paths, result/request SHA, and contract bindings. Identify native, candidate, and vendor records separately. Unreviewed historical candidates are not automatically strict Triton.','',
               '| Type | Experiment | Result | Source |','| --- | --- | --- | --- |']
        for r in row['measurements']:
            path=r['result']['path'];short=path.split('/campaigns/')[-1].split('/runs/')[-1]
            val=r['measurement'];v=f"{val['geo_mean']:.6f}×" if val.get('geo_mean') is not None else str(val['status'])
            if val.get('geo_mean') is not None and val['status']!='PASSED':v+='（'+str(val['status'])+'）'
            sources=' · '.join(local_link(evidence/s['export'],s['sha256'][:12]) for s in r.get('sources',[])) or 'Request source not located'
            role={'baseline':'Baseline directory','reference':'Vendor reference','candidate':'Candidate'}[r['role']]
            if r['constraint_status']=='known_noncompliant':role+=' / outside current constraints'
            lines.append('| '+role+' | `'+short+'` | '+v+' | '+sources+' |')
        lines+=['','## Delivery','',
                '- '+local_link(root/d['path'],'Selected source')+'：`'+d['sha256']+'`。',
                '- '+local_link(root/d['result'],'Retained full result')+f"：**{d['cann_score']:.6f}× CANN，{d['passed']}/{d['total']}**。",
                '- Remote historical measurement with the same SHA exists; connect ancestry/reference bindings segment by segment.',
                '', '## Missing connections','']+['- '+g for g in row['gaps']]
        (out/(op+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')
        pages.append(f'<tr><td><a href="{op}.md">{op}</a></td><td>{score}</td><td>{c["candidate_measurements"]} / {c["candidate_sources"]}</td><td>{d["cann_score"]:.3f}×</td><td>{d["passed"]}/{d["total"]}</td></tr>')
    head+=['','## Findings','',
           '- Native submission, optimizer first round, and regeneration starts are distinct. All 26 native submissions were located. October 21 covers eight early deliveries; October 29 covers the remaining 18. Starts use fixed batches, not lowest-score selection.',
           '- Intermediate QKV/log2/hardswish_out candidates were found; cachemask/batch_norm baselines, reference changes, and strict-Triton recovery were also found.',
           '- Not all starts have low valid scores: historical cachemask 8.651x and batch_norm 100.912x require reference-path interpretation and are separate from CANN delivery scores.',
           '- Native failures have different causes: early scatter_add relative-import failures and upsample reference-interface errors. Retain repaired baselines in full inventories.',
           '- This inventory does not establish 26 gap-free chains from 497 measurements. Connect by source SHA/parent records; reference switches are separate from optimization gains.',
           '', '## Scope and reproduction','',
           f"Search of two historical server roots found {summary['measurements']} measurements, {summary['baseline_records']} baseline-directory measurements, with {len(summary['retrieval_errors'])} parse errors. Exported sources/baselines passed SHA checks; 26 delivery sources/results match the current index.",
           '', 'Source and baseline artifacts: '+local_link(evidence/'inventory.json','Raw search index')+'; full structured inventory: [inventory.json](inventory.json).',
           '', '```powershell',
           'python -m development.tools.traces.build_lineage_inventory --evidence development/experiments/lineage/20261117/source-evidence --output development/reports/native-to-delivery-26-20261117',
           '```','']
    (out/'README.md').write_text('\n'.join(head),encoding='utf8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>26-operator start/delivery evidence</title><style>body{background:#fff7e8;color:#1b2434;font:20px/1.6 Microsoft YaHei,sans-serif;margin:40px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #d8d5c8;text-align:left}a{color:#2368b1}h1{font-size:34px}</style><h1>26 operators: ：Native start → Historical versions → Delivery</h1><p>26 baseline/delivery records located。Historical starts and CANN endpoints use different references，interpret separately。</p><p><a href="README.md">Full guide</a> · <a href="inventory.json">Structured evidence</a></p><table><tr><th>Operator</th><th>Historical native start</th><th>Candidate measurements / Source count</th><th>Delivery  CANN</th><th>Full evaluation</th></tr>'+''.join(pages)+'</table></html>',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
