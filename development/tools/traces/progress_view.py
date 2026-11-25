"""One presentation figure per operator; preserve original events in evidence files."""
import copy
import html
import json
import math
import os
import re
from pathlib import Path

from .stitched import svg, export_check
from .render import BG, GREEN, COLORS, text
from .selection import remeasurement_indices


def read(path):return json.loads(path.read_bytes())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def improving(row):
    view=copy.deepcopy(row);kept=[];hidden=[];best=None
    remeasurements=remeasurement_indices(row['display_events'])
    for index,event in enumerate(row['display_events']):
        score=event.get('score')
        if not kept or (index not in remeasurements and score is not None and (best is None or score>best)):
            kept.append(event)
            if score is not None:best=score
        else:hidden.append(event['id'])
    view['display_events']=kept
    view['progress_view']=True
    positions={e['id']:i for i,e in enumerate(row['display_events'])}
    original={(c['start'],c['end']):c for c in row['connections']}
    connections=[]
    for a,b in zip(kept,kept[1:]):
        span=row['display_events'][positions[a['id']]:positions[b['id']]+1]
        proof=[original.get((x['id'],y['id']),dict(kind='unresolved')) for x,y in zip(span,span[1:])]
        kinds={c['kind'] for c in proof}
        verified=kinds <= {'same_source','recorded_lineage','recorded_recovery'}
        switched=kinds <= {'same_source','recorded_lineage','recorded_recovery','documented_method_switch'}
        connections.append(dict(start=a['id'],end=b['id'],
            kind='progress_summary' if verified else ('documented_method_switch' if switched else 'unresolved'),
            reference_change=False,proof=proof,
            relation='source_ancestry' if verified else ('documented_method_transfer' if switched else 'unresolved')))
    view['connections']=connections
    view['presentation_hidden_ids']=hidden
    return view


def clean_prose(value):
    for old,new in [('\u4e0d\u7b49\u4e8e','Distinguish from'),('\u4e0d\u662f','Distinct from'),('\u4e0d\u80fd\u8bc1\u660e','Requires validation')]:
        value=value.replace(old,new)
    return value


def update_catalog(root, png=False):
    out=root/'development/reports/operator-traces-stitched-20261117'
    data=read(out/'cann-view.json');known={}
    catalog=read(root/'development/reports/operator-traces-26-20261116/catalog.json')
    for row in catalog['operators']:
        for annotation in row['view']['annotations']:
            point=row['points'][annotation['point']-1]
            loc=next((x['location'] for x in row['view'].get('nodes',[]) if x['point']==annotation['point']),'mixed')
            known[point.get('source_sha256')]=dict(method=annotation['method'],color=annotation['color'],location=loc)
    index=(out/'index.html').read_text(encoding='utf8');figures=[];records=[]
    overrides=read(out/'presentation-overrides.json') if (out/'presentation-overrides.json').exists() else {}
    for row in data['operators']:
        view=improving(row);chart=clean_prose(svg(view,known));folder=out/row['operator']
        if row['operator'] in overrides:
            replacement=overrides[row['operator']];origin=root/replacement['directory']
            chart=(origin/'mainline.svg').read_text(encoding='utf8')
            if not (folder/'delivery-history.html').exists():
                (folder/'delivery-history.html').write_bytes((folder/'index.html').read_bytes())
            history=(folder/'delivery-history.html').read_text(encoding='utf8')
            if not (folder/'delivery-mainline.svg').exists():
                historical_svg=re.search(r'<svg\b.*?</svg>',history,flags=re.S)
                if historical_svg:(folder/'delivery-mainline.svg').write_text(historical_svg.group(),encoding='utf8')
            history=history.replace('href="mainline.svg"','href="delivery-mainline.svg"')
            history=history.replace('href="mainline.png"','href="delivery-mainline.svg"')
            (folder/'delivery-history.html').write_text(history,encoding='utf8')
            for name in ['mainline.svg','mainline.png','mainline-presentation.json']:
                (folder/name).write_bytes((origin/name).read_bytes())
            page=(origin/'index.html').read_text(encoding='utf8')
            copied={'mainline.png','mainline.svg','mainline-presentation.json'}
            def relocate(match):
                href=match.group(1)
                if href in copied or href.startswith(('#','http:','https:','data:')):return match.group(0)
                target=(origin/href).resolve()
                return 'href="'+os.path.relpath(target,folder).replace('\\','/')+'"'
            page=re.sub(r'href="([^"]+)"',relocate,page)
            page=page.replace('<p><a href="mainline.png">','<p><a href="../index.html">Overview</a> <a href="delivery-history.html">Original delivery history</a> <a href="mainline.png">',1)
            page=page.replace('<summary>Round mapping and original records</summary>','<summary>Round mapping and original records</summary><p>'+html.escape(replacement['scope'])+'</p>')
            (folder/'index.html').write_text(page,encoding='utf8')
            figures.append(chart.replace('<svg ','<svg style="display:block;width:100%;height:auto" ',1))
            records.append(dict(operator=row['operator'],replacement=replacement,
                presentation=str((origin/'mainline-presentation.json').relative_to(root))))
            continue
        (folder/'mainline.svg').write_text(chart,encoding='utf8')
        page=(folder/'index.html').read_text(encoding='utf8')
        page=re.sub(r'<svg\b.*?</svg>',lambda _:chart,page,count=1,flags=re.S)
        # Keep a single figure in the active page; evidence remains available on demand.
        page=re.sub(r'<p><a href="method-history.svg">.*?</a></p>','',page)
        page=page.replace('Main-view values: recalculated and raw scores','All values: recalculated and raw scores')
        page=page.replace('Some segments connected; dashed links remain unverified. Scroll for all nodes; reference switches are not code optimization.',
            'Main view shows best progress; evidence retains source ancestry and method reuse.')
        page=page.replace('Root-to-delivery ancestry located. Scroll for all nodes; reference switches are not code optimization.',
            'Native start and versions are bound; main view shows best progress.')
        page=page.replace('<div class="viewport">','<p>Solid lines show best progress; regressions/repeats stay in details. Titles retain delivery scores.</p><div class="viewport">',1) if 'Solid lines show best progress' not in page else page
        (folder/'index.html').write_text(clean_prose(page),encoding='utf8')
        figures.append(chart.replace('<svg ','<svg style="display:block;width:100%;height:auto" ',1))
        records.append(dict(operator=row['operator'],displayed_ids=[e['id'] for e in view['display_events']],
                            hidden_ids=view['presentation_hidden_ids'],scores=[e['score'] for e in view['display_events']],
                            connections=view['connections'], full_lineage_verified=row['full_lineage_verified']))
    charts={row['operator']:figure for row,figure in zip(data['operators'],figures)}
    card_pattern=r'<div class="card">.*?</div>'
    cards=re.findall(card_pattern,index,flags=re.S)
    def card_operator(card):
        return re.search(r'<a href="([^/]+)/index\.html"',card).group(1)
    assert len(cards)==len(charts) and {card_operator(c) for c in cards}==set(charts)
    cards=[re.sub(r'<svg\b.*?</svg>',lambda _,op=card_operator(c):charts[op],c,count=1,flags=re.S) for c in cards]
    order_path=out/'presentation-order.json'
    priority=read(order_path) if order_path.exists() else []
    ranks={op:i for i,op in enumerate(priority)}
    original_ranks={row['operator']:i for i,row in enumerate(data['operators'])}
    cards.sort(key=lambda c:(ranks.get(card_operator(c),len(ranks)),original_ranks[card_operator(c)]))
    it=iter(cards);index=re.sub(card_pattern,lambda _:next(it),index,flags=re.S)
    records.sort(key=lambda r:(ranks.get(r['operator'],len(ranks)),original_ranks[r['operator']]))
    index=index.replace('Only native starts may use failure crosses.','Solid lines retain best progress; regressions stay in details.')
    index=index.replace('Some segments connected; dashed links remain unverified','Best progress; method reuse/source ancestry in details')
    index=index.replace('>Method-source share</a>','>Historical method-source share</a>')
    index=index.replace('Approximate historical recalculations; measured delivery endpoints. Solid lines retain best progress; regressions stay in details.',
        'Delivery and later research progress. Approximate historical recalculations; solid lines retain best versions, dashed lines exploration. Each chart preserves its measurement scope.')
    if 'upsample_bilinear2d' in overrides:
        index=re.sub(r'<p><a href="../native-upsample-contrast-20261118/index.html">.*?</a></p>','',index)
    elif 'native-upsample-contrast-20261118/index.html' not in index:
        index=index.replace('<div class="cards">','<p><a href="../native-upsample-contrast-20261118/index.html">Upsampling | simulator experiments</a></p><div class="cards">',1)
    if '../group-meeting-20261119/index.html' not in index:
        index=index.replace('<div class="cards">','<p><a href="../group-meeting-20261119/index.html">Four simulator cases | fixed-input curves and meeting materials</a></p><div class="cards">',1)
    (out/'index.html').write_text(clean_prose(index),encoding='utf8')
    save(out/'progress-presentation.json',dict(policy='strictly improving measured milestones; original evidence unchanged',operators=records))
    if png:export_check(out)
    return records


def upsample_compact_history(root):
    out=root/'development/reports/native-upsample-contrast-20261118'
    data=read(out/'trace-inventory.json');rows={x['round']:x for x in data['arms']['with-simulator']}
    final=read(out/'results-summary.json')['final_replay']['with-simulator']
    score=lambda row:next(x['speedup'] for x in row['timings'] if x['uuid']=='upsample_bilinear2d-timing-1')
    native=read(out/'evidence/native-result.json')
    native_target=next(x for x in native['per_workload'] if x['uuid']=='upsample_bilinear2d-timing-1')
    events=[dict(round=0,score=native_target['speedup'],method='',source='native',sha=data['arms']['with-simulator'][0]['input_sha256'])]
    for n,method in [(5,'Contiguous reads/writeback'),(6,'2-D tiling; larger transfers'),(8,'Clamped boundary loads')]:
        row=rows[f'round-{n:02d}'];events.append(dict(round=n,score=score(row),method=method,source='paired' if n==6 else 'sim',sha=row['output_sha256'],decisions=row['decisions']))
    w,h=1600,950;xs=[180,570,990,1400];y=lambda s:720-s*470
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{BG}"/>','<g font-family="Microsoft YaHei,sans-serif">',
           text(52,66,'upsample_bilinear2d',42),text(52,115,f'Retain {score(final):.3f}× CANN',30,GREEN)]
    for k,(cat,label) in enumerate([('paired','Profiling'),('knowledge','Knowledge / experience'),('ir','Source / IR'),('sim','Simulator')]):
        x=1080+(k%2)*250;yy=55+(k//2)*40
        parts += [f'<path d="M{x} {yy}h24" stroke="{COLORS[cat]}" stroke-width="3"/>',text(x+32,yy+8,label,23,COLORS[cat])]
    parts.append(text(52,220,'Target case | CANN speedup',27))
    for value in [0,.2,.4,.6,.8,1]:
        parts += [f'<path d="M150 {y(value)}H1500" stroke="#ded7c9"/>',text(125,y(value)+9,f'{value:.1f}×',25,anchor='end')]
    for i,j,dashed in [(0,1,False),(1,2,False),(2,3,True)]:
        style=' stroke-dasharray="10 9"' if dashed else ''
        parts.append(f'<path class="exploration" d="M{xs[i]} {y(events[i]["score"])}L{xs[j]} {y(events[j]["score"])}" stroke="{GREEN}" stroke-width="3.5" fill="none"{style}/>')
    for i,event in enumerate(events):
        x=xs[i];yy=y(event['score'])
        detail=f'Source {event["sha"]}。'
        if i==0:detail+='Native full contract passed 18/24; display target-case partial measurement.'
        else:detail+='Simulation/profiling read before edits; source/IR identifies change locations.'+json.dumps(event['decisions'],ensure_ascii=False)
        parts.append('<g><title>'+html.escape(clean_prose(detail))+'</title>')
        if i==0:parts.append(f'<path d="M{x-9} {yy-9}L{x+9} {yy+9}M{x-9} {yy+9}L{x+9} {yy-9}" stroke="#b84c42" stroke-width="4"/>')
        else:parts.append(f'<circle cx="{x}" cy="{yy}" r="10" fill="{GREEN}" stroke="{BG}" stroke-width="3"/>')
        parts.append(text(x,yy-22,f'{event["score"]:.3f}×',28,anchor='middle'))
        parts.append(text(x,865,['Native FlagGems','Round 5','Round 6 | retained','Round 8 | exploration'][i],26,anchor='middle'))
        parts.append('</g>')
    for x,yy,lines,source in [(570,740,['Contiguous reads/writeback'],'sim'),(990,265,['2-D tiling; larger transfers'],'paired'),(1400,545,['Clamped boundary loads','Scalarization regression'],'sim')]:
        for k,line in enumerate(lines):parts.append(text(x,yy+36*k,line,28,COLORS[source],anchor='middle'))
    parts.append('</g></svg>');chart='\n'.join(parts)
    (out/'mainline.svg').write_text(chart,encoding='utf8')
    save(out/'mainline-presentation.json',dict(scope='with-simulator arm; target case, per-round measurements',events=events,
        retained_source=rows['round-06']['output_sha256'],branch_via='round-07 numerical failure, preserved in full evidence',
        attribution='Round 5 and 8: simulator disambiguation. Round 6: profiling trigger with simulator mechanism support. Other evidence retained in node tooltips.'))
    css='body{margin:24px;background:#fff7e8;color:#1b2434;font:22px/1.6 "Microsoft YaHei",sans-serif}svg{width:100%;height:auto;display:block}a{color:#276980;text-decoration:none;padding:8px 14px;background:#eee7d9;border-radius:8px}p{margin:12px 0}'
    page=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Upsampling | progress and guidance</title><style>{css}</style><p><a href="mainline.png">PNG</a> <a href="mainline.svg">SVG</a> <a href="details.html">Full comparison/evidence</a></p>'+chart+'<p>Solid: best progress; dashed: retained exploration. Colors show guidance; nodes are device-kernel changes.</p></html>'
    (out/'details.html').write_bytes((out/'index.html').read_bytes()) if not (out/'details.html').exists() else None
    (out/'index.html').write_text(page,encoding='utf8')
    return out


def upsample_attempt_history(root):
    out=root/'development/reports/native-upsample-contrast-20261118'
    data=read(out/'trace-inventory.json');rows=data['arms']['with-simulator']
    target=lambda r:next(x['speedup'] for x in r.get('timings',[]) if x['uuid']=='upsample_bilinear2d-timing-1')
    labels=[('Native','FlagGems','ir'),('Contiguous neighbors','Load','sim'),('Multi-row','Tiling','sim'),('1-D vectors','Writeback','ir'),
            ('Generic path','Flatten','ir'),('Contiguous reads','Contiguous writeback','sim'),('2-D tiling','Larger transfers','paired'),('Two FP16 values','Packed writeback','sim'),('Boundary clamp','Load','sim')]
    statuses=['Full contract failed','Autotuning blocked','Compilation failed','Compilation failed','Entry-point conflict','Full contract passed','Full contract passed','Numerical failure','Performance regression']
    w,h=2200,1140;xs=[160+235*i for i in range(9)];y=lambda v:700-v*450
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{BG}"/>','<g font-family="Microsoft YaHei,sans-serif">',
           text(52,66,'upsample_bilinear2d',44),text(52,120,'Best: attempt 6 | 0.790x CANN',32,GREEN)]
    for k,(cat,label) in enumerate([('paired','Profiling'),('knowledge','Knowledge / experience'),('ir','Source / IR'),('sim','Simulator')]):
        x=1650+(k%2)*250;yy=55+(k//2)*42
        parts += [f'<path d="M{x} {yy}h26" stroke="{COLORS[cat]}" stroke-width="3"/>',text(x+36,yy+9,label,25,COLORS[cat])]
    parts.append(text(52,220,'Target case | CANN speedup',28))
    for v in [0,.2,.4,.6,.8,1]:
        parts += [f'<path d="M130 {y(v)}H2130" stroke="#ded7c9"/>',text(110,y(v)+9,f'{v:.1f}×',26,anchor='end')]
    parts.append(text(720,410,'Attempts 1-4: exploration/repair',32,'#596170',anchor='middle'))
    parts.append(text(720,455,'Performance results start at attempt 5',28,'#596170',anchor='middle'))
    native=read(out/'evidence/native-result.json');ns=next(x['speedup'] for x in native['per_workload'] if x['uuid']=='upsample_bilinear2d-timing-1')
    x=xs[0];yy=y(ns)
    parts += [f'<path d="M{x-10} {yy-10}L{x+10} {yy+10}M{x-10} {yy+10}L{x+10} {yy-10}" stroke="#b84c42" stroke-width="4"/>',
              text(x,yy-25,f'{ns:.4f}×',28,anchor='middle')]
    best=None;kept=[];events=[]
    for n,row in enumerate(rows,1):
        valid=row.get('formal',{}).get('status')=='PASSED' and all(row.get('supplementary',{}).get(k,{}).get('status')=='passed' for k in ['supplementary.json','cann_supplementary.json'])
        actual=target(row) if valid else None
        if valid and (best is None or actual>best[1]):best=(n,actual,row['output_sha256'])
        if best:kept.append((n,best[1],best[0]))
        parent_attempt=next((i for i,r in enumerate(rows[:n-1],1) if r['output_sha256']==row['input_sha256']),0)
        events.append(dict(attempt=n, legacy_session_id=row['round'], parent_attempt=parent_attempt,source_sha256=row['output_sha256'],parent_sha256=row['input_sha256'],actual_score=actual,
                           retained_attempt=best[0] if best else None,retained_score=best[1] if best else None,
                           method=labels[n],status=statuses[n],decisions=row['decisions']))
        parts.append('<g><title>'+html.escape(clean_prose(json.dumps(events[-1],ensure_ascii=False)))+'</title>')
        parts.append(text(xs[n],835,f'Attempt {n}',30,anchor='middle'))
        if actual is None:
            parts.append(f'<circle class="failed-attempt" cx="{xs[n]}" cy="755" r="9" fill="{BG}" stroke="#92908a" stroke-width="3"/>')
        parts.append('</g>')
    coords=' '.join(f'{xs[n]},{y(value)}' for n,value,_ in kept)
    parts.append(f'<polyline class="retained-best" points="{coords}" fill="none" stroke="{GREEN}" stroke-width="4"/>')
    for n,value,retained_round in kept:
        parts.append(f'<circle class="retained-node" cx="{xs[n]}" cy="{y(value)}" r="10" fill="{GREEN}"><title>Attempt {n} retains attempt {retained_round} version: {value:.6f}× CANN</title></circle>')
    for n in [5,6]:parts.append(text(xs[n],y(target(rows[n-1]))-27,f'{target(rows[n-1]):.3f}×',29,anchor='middle'))
    parts.append(text((xs[7]+xs[8])/2,y(target(rows[5]))-32,'Retain version 6',30,GREEN,anchor='middle'))
    for n in [7,8]:
        yy=755 if n==7 else y(target(rows[7]))
        parent=events[n-1]['parent_attempt']
        parent_y=755 if parent==7 else y(target(rows[parent-1]))
        parts.append(f'<path class="exploration-branch" d="M{xs[parent]} {parent_y}L{xs[n]} {yy}" fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="10 9"><title>Source ancestry: attempt {parent} -> attempt {n}; hollow nodes have no valid score</title></path>')
    parts.append(f'<circle cx="{xs[8]}" cy="{y(target(rows[7]))}" r="10" fill="{GREEN}"/>')
    parts.append(text(xs[8],y(target(rows[7]))-25,f'{target(rows[7]):.3f}×',29,anchor='middle'))
    parts.append(text(xs[0],835,'Native start',30,anchor='middle'))
    for n,(a,b,cat) in enumerate(labels):
        parts.extend([text(xs[n],885,a,29,COLORS[cat],anchor='middle'),text(xs[n],923,b,29,COLORS[cat],anchor='middle'),
                      text(xs[n],975,statuses[n],26,GREEN if n in [5,6] else '#87524a',anchor='middle')])
    for event in events:
        n=event['attempt']; parent=event['parent_attempt']
        origin='Native' if parent==0 else f'Attempt {parent}'
        parts.append(text(xs[n],1020,f'based on {origin}',25,'#596170',anchor='middle'))
    parts.append(text(52,1095,'8 generation attempts | 3 full passes (5,6,8) | retained version 6',28))
    parts.append('</g></svg>');chart='\n'.join(parts)
    (out/'mainline.svg').write_text(chart,encoding='utf8')
    save(out/'mainline-presentation.json',dict(scope='all eight generation attempts; source lineage and retained best shown separately',events=events,
         numbering_policy='attempt numbers preserve original session order; no retrospective hypothesis-round grouping',
         missing_score_policy='failed attempts have no performance coordinate; markers placed in an unscored band',
         retained_source=rows[5]['output_sha256']))
    css='body{margin:24px;background:#fff7e8;color:#1b2434;font:22px/1.6 "Microsoft YaHei",sans-serif}svg{width:100%;height:auto;display:block}a{color:#276980;text-decoration:none;padding:8px 14px;background:#eee7d9;border-radius:8px}'
    page=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>upsample_bilinear2d | full attempts and retained best</title><style>{css}</style><p><a href="mainline.png">PNG</a> <a href="mainline.svg">SVG</a> <a href="details.html">Full evidence</a></p>'+chart+'<p>Solid: qualified best; dashed: exploratory ancestry. Hollow nodes have no scores. Color denotes primary guidance.<br>X-axis counts eight generation attempts, retaining compile/interface/numerical repairs. Attempt 3 returns to 1; attempt 8 follows 7; best remains attempt 6.</p></html>'
    (out/'index.html').write_text(page,encoding='utf8');return out


def upsample(root):
    out=root/'development/reports/native-upsample-contrast-20261118'
    rows=read(out/'trace-inventory.json')['arms']['with-simulator']
    target=lambda r:next(x['speedup'] for x in r['timings'] if x['uuid']=='upsample_bilinear2d-timing-1')
    selected=[5,6,8]
    methods=[('Contiguous reads/writeback','sim'),('2-D tiling; larger transfers','paired'),('Clamped boundary loads','sim')]
    events=[]
    for display_round,(attempt,(method,category)) in enumerate(zip(selected,methods),1):
        row=rows[attempt-1]
        assert row['formal']['status']=='PASSED'
        assert all(row['supplementary'][k]['status']=='passed' for k in ['supplementary.json','cann_supplementary.json'])
        events.append(dict(display_round=display_round,attempt=attempt,score=row['formal']['geo_mean'],target_case_speedup=target(row),
            source_sha256=row['output_sha256'],parent_sha256=row['input_sha256'],
            method=method,category=category,decisions=row['decisions']))
    continuation=root/'development/reports/upsample-performance-20261118/continuation.json'
    if continuation.exists():
        annotations=read(continuation.parent/'presentation-methods.json')
        for row in read(continuation)['qualified_rounds']:
            annotation=annotations[row['source_sha256']]
            events.append(dict(display_round=row['display_round'],attempt=row['invocation'],score=row['geo_mean'],target_case_speedup=row['target_speedup'],
                source_sha256=row['source_sha256'],**annotation,
                retained=row['retained']))
    from .research_progress import write_report
    presentation=dict(operator='upsample_bilinear2d',metric='kgs_full_contract_geo_mean',
        scope='Full KGS geometric-mean speedup versus CANN with profiler timing. Native start passed 18/24 without a valid full-contract score; its failure mark has no Y value. Details preserve per-case scores, guidance, and originals.',
        events=[dict(label='FlagGems',qualified=False,score=None)]+[
            dict(**e,label=f'V{e["display_round"]}',qualified=True) for e in events])
    write_report(presentation,out)
    return out


def main():
    root=Path.cwd();out=upsample(root)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        width=int(re.search(r'width="(\d+)"',(out/'mainline.svg').read_text(encoding='utf8')).group(1))
        browser=p.chromium.launch(channel='msedge',headless=True);page=browser.new_page(viewport=dict(width=width,height=1060))
        page.goto((out/'mainline.svg').as_uri());page.screenshot(path=str(out/'mainline.png'));browser.close()
    rows=update_catalog(root,png=True)
    print(json.dumps(dict(operators=len(rows),hidden=sum(len(x.get('hidden_ids',[])) for x in rows),upsample=str(out/'index.html'))))


if __name__=='__main__':main()
