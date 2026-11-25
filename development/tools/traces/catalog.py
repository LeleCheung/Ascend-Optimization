"""Generate a 26-operator atlas from deliveries and frozen ledgers without inventing missing rounds."""
import argparse
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .importer import digest, read
from .index import save
from .render import draw, CSS


# Display selections and short English methods; catalog.json retains original plans, sources, and scores.
SELECTIONS = {
 'selu_':('c3e781', [('Direct launch; simpler host path','host','probe'),('Fixed-grid streaming loop','device','probe')]),
 'rad2deg_':('b20dea', [('Prebind compiled launcher','host','probe'),('Size-dependent tiling','device','probe'),('Preserve input precision','device','ir')]),
 'uniform':('20d459', [('Vectorize Philox/state access','mixed','probe'),('Cache direct compiled launch','host','paired')]),
 'value_selecting_reduction_backward':('d9428d', [('Fixed-grid streaming gradient writeback','mixed','unknown')]),
 'scatter_add':('42950a', [('Sorted segmented reduction','device','knowledge'),('Move lookup on chip','device','probe')]),
 'upsample_bilinear2d':('c34839', [('On-chip gather replaces misaligned global reads','device','knowledge'),('Cache shapes/launch configuration','host','knowledge'),('Remove redundant masks: no gain','device','probe')]),
 '_conj_physical':('62faff',[('Prebind compiled launcher','host','probe'),('Simplify output allocation','host','probe'),('Add safe fallback: regression','host','unknown')]),
 'negative_':('033b6a',[('Direct compiled-kernel launch','host','probe'),('64 KB per program','device','paired')]),
 'xlogy_scalar_tensor':('016df1',[('Cache launch plan/output allocation','host','probe')]),
 'nan_to_num_':('0af95a',[('Simplify launch arguments','host','probe'),('Separate small/large-input paths','mixed','probe'),('Single-stage multi-tile loop','device','probe'),('Original-precision experiment: no gain','device','ir')]),
 'cross_out':('023367',[('Reorganize indexing/neighbors','device','probe'),('Larger resident tiles','device','probe'),('Simplify launch/repair boundaries','mixed','probe'),('Larger 32-bit tiles','device','ir')]),
 'hardshrink_backward':('e0a716',[('Fixed-grid unmasked loop','device','probe'),('Simplify launch/single-tile specialization','mixed','probe'),('Tail-only boundary handling','device','probe')]),
 'subtract_':('dcd20c',[('Direct compiled-kernel launch','host','probe')]),
 'native_group_norm':('092021',[('Cache launch/combine allocation','host','unknown'),('Fewer launch arguments: regression','host','probe')]),
 'log2_':('ddeb4e',[]),
 'fix':('a2d5ed',[('Fixed-grid multi-tile loop: regression','device','probe'),('Adjust pipeline/launch fallback','mixed','probe'),('Repair launch argument count','host','ir')]),
 'round_out':('009743',[('Streaming loop/simpler rounding','device','probe'),('Two-stage pipeline experiment','device','probe'),('Larger 16-bit tiles: no gain','device','probe')]),
 'special_round_out':('cec136',[('Cache direct launch path','host','probe'),('Simplify FP32 rounding','device','ir'),('16-bit rounding: failure','device','ir')]),
 'special_xlogy_':('a754ce',[('Cache grid/launch arguments','host','probe'),('Simplify host lookup/queries','host','probe')]),
 'hardswish':('1c3071',[('Unmask contiguous tiles','device','probe'),('Reduce small-input launch cost','host','paired'),('Original-precision vector math: regression','device','ir')]),
 'fmod_tensor_':('aca31f',[('Simplify host metadata','host','probe')]),
}


def make_catalog(root, output):
    delivery = read(root/'operators/index.json')
    matrix_path = root/'development/reports/trace-attribution-matrix-20261116.json'
    matrix = {r['operator']:r for r in read(matrix_path)['rows']}
    accepted = [r for r in delivery['rows'] if r['scope']=='main_30' and r['accepted']]
    rows=[]
    for selected in accepted:
        op=selected['operator']; archived=matrix[op]
        if op=='hardswish_out':
            bases=[root/'development/experiments/regressions/20261109'/batch/'operators'/op
                   for batch in ('fixed-25','hardswish-native-low-precision')]
            requests=[read(b/'evaluation/request.json') for b in bases]
            assert requests[0]['binding']==requests[1]['binding']
            assert requests[0].get('settings')==requests[1].get('settings')
            reviews=[read(b/'review.json') for b in bases]
            assert reviews[1]['parent_source_sha256']==reviews[0]['source_sha256']
            points=[]
            for n,(base,req,review) in enumerate(zip(bases,requests,reviews)):
                source=(base/'candidate.py').read_bytes()
                assert digest(source)==review['source_sha256']
                assert req['implementation']['sources'][0]['content'].encode('utf8')==source
                result=read(base/'evaluation/result.json')
                points.append(dict(id=op+':repair-'+str(n),score=result['geo_mean'],source_sha256=digest(source),
                    method='Before repair' if n==0 else 'Adjust low-precision math/tiling',
                    result=(base/'evaluation/result.json').relative_to(root).as_posix(),review=review))
            view=dict(id='mainline',session=op,title=op+' optimization trace',subtitle='Relative to CANN | same-contract repair',ylim=[0,1.1],
                annotations=[dict(point=2,method='Adjust low-precision math/tiling',color='ir',position=[750,285],arrow=[1200,325])],
                nodes=[dict(point=1,location='mixed'),dict(point=2,location='device')],
                line_status='Below: delivery repair')
            record=dict(operator=op,kind='verified_delivery_repair',points=points,view=view,
                        history=archived['historical_rounds'],gap=None,selected_source=[str(b.relative_to(root)) for b in bases])
        elif op=='log10_':
            pilot=read(root/'development/manifests/traces/log10_/index.json')
            view=next(v for v in pilot['views'] if v['id']=='mainline')
            points=[p for p in pilot['evaluations'] if p['session_id']==view['session']]
            svg=draw(pilot,view)
            record=dict(operator=op,kind='recovered_continuous_trace',selected_source='development/manifests/traces/log10_/index.json',
                points=points,view=view,history=archived['historical_rounds'],gap=None)
        elif op in SELECTIONS:
            prefix,labels=SELECTIONS[op]
            points_raw=[r for r in archived['historical_rounds'] if r['ledger_sha256'].startswith(prefix)]
            assert len(points_raw)==len(labels)+1,(op,len(points_raw),len(labels))
            points=[]
            for n,p in enumerate(points_raw):
                points.append(dict(id=op+':round-'+str(p['round']),score=p['evaluation'].get('geo_mean'),
                    source_sha256=p.get('source_sha256'),method='Batch start' if n==0 else labels[n-1][0],raw=p))
            high=max(p['score'] or 0 for p in points)*1.16
            high=max(.5,math.ceil(high*10)/10)
            annotations=[]
            # At most four method arrows; template positions do not replace original method records.
            for i,label in enumerate(labels[:4],2):
                placements={1:([220,285],[540,325]),2:([930,285],[1210,325]),3:([220,775],[540,730]),4:([930,775],[1200,730])}
                if len(labels)==1:position,arrow=[550,285],[950,325]
                else:position,arrow=placements[i-1]
                annotations.append(dict(point=i,method=label[0],color=label[2],position=position,arrow=arrow))
            legend=['paired','sim','knowledge','probe','ir']
            if any(a['color']=='unknown' for a in annotations):legend=['paired','sim','knowledge','probe','unknown']
            gap='Only one formal measurement located; optimization history incomplete' if len(points)==1 else None
            view=dict(id='mainline',session=op,title=op+' optimization trace',subtitle='Historical KGS reference | batch '+str(len(points))+' formal evaluations',
                ylim=[0,high],annotations=annotations,legend=legend,gap=gap,x_label='Formal evaluation order in this batch',plot_bottom=650 if len(labels)>2 else 745,
                nodes=[dict(point=1,location='mixed')]+[dict(point=i,location=label[1]) for i,label in enumerate(labels,2)])
            if op=='_conj_physical':view['subtitle']='Historical reference may include CPU fallback | batch '+str(len(points))+' formal evaluations'
            svg=draw(dict(evaluations=[dict(p,session_id=op) for p in points],sessions=[dict(id=op,counts={})]),view)
            ledger=next(l for l in archived['ledger_files'] if l['sha256'].startswith(prefix))
            record=dict(operator=op,kind='saved_ledger_attempt_order',selected_source=ledger,points=points,view=view,
                history=archived['historical_rounds'],gap=gap,
                relation_note='Show formal rounds from one original ledger; unverified edit ancestry remains unresolved.')
        else:
            result=read(root/'operators'/op/'evidence/retained/evaluation/result.json')
            points=[dict(id=op+':retained-evaluation',score=result['geo_mean'],method='Delivered-source regression',source_sha256=archived['delivery_source_sha256'])]
            gap='Delivery results retained; optimization rounds incomplete'
            view=dict(id='mainline',session=op,title=op+' record gap',subtitle='Relative to CANN | retained measurement',
                ylim=[0,max(1.,math.ceil(result['geo_mean']*1.3*10)/10)],annotations=[],gap=gap,
                x_label='Retained measurement (not optimization history)',nodes=[dict(point=1,location='mixed')])
            svg=draw(dict(evaluations=[dict(p,session_id=op) for p in points],sessions=[dict(id=op,counts={})]),view)
            record=dict(operator=op,kind='delivery_snapshot_only',points=points,view=view,history=[],gap=gap,
                selected_source='operators/'+op+'/evidence/retained/evaluation/result.json')
        manifest=read(root/selected['manifest'] if selected['manifest'].startswith('operators/') else root/'operators'/selected['manifest'])
        source=root/manifest['selected']['path']
        assert digest(source.read_bytes())==manifest['selected']['sha256']
        result_path=root/'operators'/op/'evidence/retained/evaluation/result.json'
        result=read(result_path)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']
        assert abs(result['geo_mean']-selected['cann_score'])<1e-10 and selected['cann_score']>=.8
        record['delivery']=dict(score=selected['cann_score'],source=manifest['selected']['path'],
            source_sha256=manifest['selected']['sha256'],passed=result['num_passed'],total=result['num_workloads'],
            result=result_path.relative_to(root).as_posix(),result_sha256=digest(result_path.read_bytes()),
            endpoint_same_source=points[-1].get('source_sha256')==manifest['selected']['sha256'])
        view=record['view']
        view['delivery']=record['delivery']
        view['result_only']=len(points)==1
        if view['result_only']:view['title']=op+' delivery result'
        elif not view.get('line_status'):
            view['line_status']='Below: independent regeneration' if op=='log10_' else 'Below: historical optimization segment'
        svg=draw(dict(evaluations=[dict(p,session_id=view['session']) for p in points],sessions=[dict(id=view['session'],counts={})]),view)
        directory=output/op;directory.mkdir(parents=True,exist_ok=True)
        (directory/'mainline.svg').write_text(svg,encoding='utf8')
        record.update(current_cann_score=selected['cann_score'],historical_simulator_files=len(archived['simulator_files']))
        if op=='_conj_physical':
            record['reference_limitation']='Historical references may include CPU fallback; large ratios are separate from native-CANN speedup and current CANN scores.'
        save(directory/'trace.json',record)
        details=html.escape(json.dumps(record,ensure_ascii=False,indent=2))
        (directory/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>'+CSS+'</style><main>'
            '<p><a href="../index.html">26-operator overview</a> | <a href="mainline.png">PNG</a> | <a href="mainline.svg">SVG</a> | <a href="trace.json">Trace data</a></p>'
            +'<figure>'+svg+'</figure><details><summary>Measurement scope and legend</summary><p>'+html.escape(record['view']['subtitle'])+'</p><p>Y-axis is reference time divided by candidate time under original KGS statistics. X-axis numbers displayed evaluated versions, not all model calls. Profiling includes paired collection, host profiling, and probes; code analysis records source/IR references without excluding other tools. Colors retain primary recorded categories, not independent causal contributions.</p></details>'
            +'<p><a href="../../../../'+html.escape(record['delivery']['source'])+'">Current delivered source</a> | <a href="../../../../'+html.escape(record['delivery']['result'])+'">Delivery artifacts</a></p>'
            +('<p class="note">'+html.escape(record['gap'])+'</p>' if record['gap'] else '')
            +('<p class="note">'+html.escape(record['reference_limitation'])+'</p>' if record.get('reference_limitation') else '')
            +'<details><summary>Selection, collected history, and original plans</summary><pre>'+details+'</pre></details></main></html>',encoding='utf8')
        rows.append(record)
    assert len(rows)==26
    result=dict(schema='hgkda.trace-catalog/v1',operators=rows,source=dict(path=matrix_path.relative_to(root).as_posix(),sha256=digest(matrix_path.read_bytes())),
        counts=dict(total=26,multinode=sum(len(r['points'])>1 for r in rows),single_ledger=sum(r['kind']=='saved_ledger_attempt_order' and len(r['points'])==1 for r in rows),snapshot_only=sum(r['kind']=='delivery_snapshot_only' for r in rows)))
    save(output/'catalog.json',result)
    cards=[]
    for r in rows:
        op=r['operator']
        cards.append('<article><h2><a href="'+op+'/index.html">'+html.escape(op)+'</a></h2><a href="'+op+'/index.html"><img loading="lazy" src="'+op+'/mainline.svg"></a><p>'
            +('Optimization history incomplete' if r['gap'] else '<a href="'+op+'/index.html">Detailed records</a>')+'</p></article>')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>26-operator traces</title><style>'+CSS+
        'section{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:24px}article{border:1px solid #cec7b8;padding:12px;border-radius:8px}img{width:100%}article h2{font-size:24px;overflow-wrap:anywhere}</style><main><h1>26-operator traces</h1><p>One main sequence per operator; open complete records for details.</p><p><a href="catalog.json">Trace data</a> | <a href="../operator-traces-history-20261116/index.html">More historical traces</a></p><section>'+''.join(cards)+'</section></main></html>',encoding='utf8')
    notes=['# 26-operator trace atlas','',
           '[Open atlas](index.html) | [Full data](catalog.json) | [Layout/link checks](presentation-check.json)','',
           '22 operators have multiple displayed nodes; four have delivery-result cards. hardswish_out has two same-contract repair points. log2_ retains its early single point in data while showing the delivery result.','',
           'Each chart selects a main attempt from one ledger; log10 uses recovered ancestry. Attempt order is separate from direct source inheritance. Keep references/contracts/starts/batches distinct; HTML details and trace.json retain all collected history/plans. Display counts do not equal total optimization rounds.','',
           'Arrow colors show recorded guidance and node shapes show changed objects. Unknown provenance remains unverified. Historical categories, model statements, and tool co-occurrence do not prove independent causality. Keep missing-history/reference scope in details.','',
           '| Operator | Main nodes | Collected records | Scope |','| --- | ---: | ---: | --- |']
    for row in rows:
        notes.append('| ['+row['operator']+']('+row['operator']+'/index.html) | '+str(len(row['points']))+' | '+str(len(row['history']))+' | '+(row.get('gap') or 'Selection and historical provenance retained')+' |')
    notes+=['','## Rebuild','',
            '```powershell','python -m development.tools.traces.catalog --output development/reports/operator-traces-26-20261116 --png','```','',
            'SVG/JSON use the standard library; PNG/layout checks optionally use Playwright/Edge. Compact 24-pixel top-right legends show used categories only. Details/JSON retain reference scopes/counts. Profiling combines paired collection, host profiling, and probes without rewriting original categories. Avoid annotation overlap and step-shaped curves.','']
    (output/'README.md').write_text('\n'.join(notes),encoding='utf8')
    print(json.dumps(result['counts']))


def export_and_check(output):
    """Export 26 PNGs and check text boundaries, overlapping annotations, and local links."""
    from playwright.sync_api import sync_playwright
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport=dict(width=1600,height=900))
            for path in sorted(output.glob('*/mainline.svg')):
                page.goto(path.as_uri());page.screenshot(path=str(path.with_suffix('.png')))
                issues=page.locator('svg').evaluate('''svg=>{
                  const bad=[...svg.querySelectorAll('text')].filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>1601||b.y<0||b.y+b.height>901}).map(e=>'overflow: '+e.textContent);
                  const texts=[...svg.querySelectorAll('.method-annotation text')];
                  for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
                    const a=texts[i].getBBox(),b=texts[j].getBBox();
                    if(a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height)bad.push('method overlap: '+texts[i].textContent+' / '+texts[j].textContent)
                  }return bad}''')
                checks.append(dict(operator=path.parent.name,issues=issues))
        finally:browser.close()
    missing=[]
    for doc in output.rglob('*.html'):
        for target in re.findall(r'(?:href|src)="([^"]+)"',doc.read_text(encoding='utf8')):
            url=urlsplit(target)
            if not url.scheme and url.path and not (doc.parent/unquote(url.path)).resolve().exists():
                missing.append(dict(file=doc.relative_to(output).as_posix(),target=target))
    result=dict(operators=len(checks),pngs=len(list(output.glob('*/mainline.png'))),
                issues=[r for r in checks if r['issues']],missing_links=missing)
    save(output/'presentation-check.json',result)
    assert len(checks)==26 and not result['issues'] and not missing,result
    print(json.dumps(result,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    p.add_argument('--png',action='store_true',help='Export and validate PNG with optional Playwright/Edge')
    a=p.parse_args();make_catalog(a.root.resolve(),a.output.resolve())
    if a.png:export_and_check(a.output.resolve())


if __name__=='__main__':main()
