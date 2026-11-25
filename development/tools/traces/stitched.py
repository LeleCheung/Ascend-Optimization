"""Connect history by source identity and explicit parents; reference switches are separate from optimization gains."""
import argparse
import ast
import collections
import copy
from functools import lru_cache
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path

from .render import BG, GREEN, INK, COLORS, text


def read(p):return json.loads(p.read_bytes())
def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def campaign(path):
    for token in ('/campaigns/','/runs/'):
        if token in path:return path.split(token,1)[1].split('/')[0]
    return 'unknown'


def reference(m,bundles,linkroot):
    p=m['result']['path'].lower(); name=(m.get('implementation',{}).get('name') or '').lower()
    if '/pytorch/' in p or 'pytorch' in name:return 'PyTorch','Explicit PyTorch evaluation directory/request name'
    if '/cann/' in p or 'vs-cann' in name or 'strict-cann' in name:return 'CANN','Explicit CANN evaluation directory/request name'
    if '/cann-priority-' in p and '/comparisons/' in p:return 'CANN','CANN-first batch comparison'
    if '/original/' in p:return 'PyTorch','Original-reference evaluation directory'
    binding=m.get('contract',{}).get('binding') or {}
    bundle=bundles.get(binding.get('bundle_id'))
    if bundle:
        for member in bundle['members']:
            if member['member'].endswith('oracle.py'):
                src=(linkroot/member['export']).read_text(encoding='utf8')
                if 'Independent PyTorch reference' in src or ('import torch' in src and 'CANN' not in src and 'ctypes' not in src):
                    return 'PyTorch','PyTorch oracle bound in contract archive'
    return 'Historical reference','Reference implementation not separately verified'


def order(m):
    p=m['result']['path'];c=campaign(p);d=re.search(r'20\d{6}',c)
    date=d.group() if d else '99999999'
    phase=0 if m.get('baseline_path') else 1
    # For same-day branches this is a display grouping, never a parent claim.
    ranks={'paired-pointwise':1,'buffer-pointwise':2,'endurance':3,'next25':1,'remaining-nine':2,'threeway':3,'cann-priority':4,'strict-triton':5,'strict-night':6,'strict-five':7,'strict-next':8}
    rank=next((v for k,v in ranks.items() if c.startswith(k)),1)
    phase=0 if m.get('baseline_path') else (4 if 'codex-disjoint' in p else (1 if 'codex-late' in p else 2))
    return (date,rank,c,phase,p)


def build(row, evidence, linkroot, linkdata):
    corrections_path=Path('development/experiments/lineage/20261117/baseline-corrections.json')
    correction=read(corrections_path).get(row['operator']) if corrections_path.exists() else None
    if correction:
        row=copy.deepcopy(row)
        baseline=next(m for m in row['measurements'] if m['result']['sha256']==correction['result_sha256'])
        assert any(s['sha256']==correction['source_sha256'] for s in baseline['sources'])
        upstream=Path(correction['upstream_path'])
        assert digest(upstream.read_bytes())==correction['upstream_sha256']
        plan=next(d for d in linkdata['documents'] if d['path'].endswith(correction['plan_suffix']))
        record=next(r for r in plan['data']['records'] if r['operator']==row['operator'])
        assert record['source_sha256']==correction['source_sha256']
        assert record['upstream_source_sha256']==correction['upstream_sha256']
        row['baseline']=baseline
        row['upstream'].update(path=str(upstream),sha256=correction['upstream_sha256'],plan={})
    op=row['operator'];baseline=row['baseline'];delivery=row['delivery']
    docs={x['path']:x for x in linkdata['documents']}
    links={x['result_path']:x for x in linkdata['records'] if x['operator']==op}
    bundles={x['bundle_id']:x for x in linkdata['bundles']}
    sources={}
    for m in row['measurements']:
        for s in m.get('sources',[]):sources[s['sha256']]=evidence/s['export']
    for s in linkdata.get('sources',[]):sources[s['sha256']]=linkroot/s['export']
    for source in Path('development/experiments/lineage/20261117/extra-sources').glob('*.py'):
        assert digest(source.read_bytes())==source.stem
        sources[source.stem]=source
    seed_file=row['upstream']['plan'].get('seed',{})
    if seed_file.get('export'):sources[seed_file['sha256']]=evidence/seed_file['export']
    keys={};core=ast.parse(Path(row['upstream']['path']).read_text(encoding='utf8')).body
    core_dump=[ast.dump(x) for x in core]
    for sha,path in sources.items():
        assert digest(path.read_bytes())==sha
        if path.suffix!='.py':keys[sha]=sha;continue
        try:
            nodes=ast.parse(path.read_text(encoding='utf-8-sig')).body
            same_prefix=len(nodes)>=len(core) and [ast.dump(x) for x in nodes[:len(core)]]==core_dump
            tail=nodes[len(core):]
            # Only simple entry aliases and a one-return wrapper qualify as adaptation.
            simple=all((isinstance(x,ast.Assign) and isinstance(x.value,ast.Name)) or
                       (isinstance(x,ast.FunctionDef) and x.name=='run' and len(x.body)==1 and isinstance(x.body[0],ast.Return)
                        and isinstance(x.body[0].value,ast.Call) and isinstance(x.body[0].value.func,ast.Name)) for x in tail)
            keys[sha]='native:'+op if same_prefix and simple else 'ast:'+digest(ast.dump(ast.Module(body=nodes,type_ignores=[])).encode())
        except (SyntaxError,UnicodeError):keys[sha]=sha
    key=lambda s:keys.get(s,s)
    edges=[];edge_pairs=set();methods={}
    def edge(a,b,kind,proof,note=''):
        if not a or not b or a==b:return
        a,b=key(a),key(b)
        if a==b:return
        if (a,b,kind) in edge_pairs:return
        edge_pairs.add((a,b,kind));edges.append(dict(parent=a,child=b,kind=kind,evidence=proof,note=note))
    opdocs={p for m in links.values() for p in m['documents']}
    def nested_parent_records(obj):
        if isinstance(obj,list):
            for value in obj:yield from nested_parent_records(value)
        elif isinstance(obj,dict):
            parent=(obj.get('parent_source_sha256') or obj.get('parent_sha256')
                    or obj.get('parent_generated_source_sha256'))
            child=obj.get('source_sha256')
            if isinstance(parent,str) and isinstance(child,str):yield parent,child
            for value in obj.values():
                if isinstance(value,(dict,list)):yield from nested_parent_records(value)
    # Unmeasured intermediate proposals can carry parent identities even when
    # their documents are not attached to a final evaluation record.
    for path in docs:
        doc=docs[path];obj=doc['data']
        for parent,child in nested_parent_records(obj):
            if parent in keys and child in keys:edge(parent,child,'explicit_nested_parent',path)
        if not isinstance(obj,dict):continue
        child=obj.get('source_sha256');parent=(obj.get('parent_source_sha256') or obj.get('origin_sha256')
                                               or obj.get('parent_sha256') or obj.get('reviewed_parent_source_sha256'))
        origin=obj.get('origin')
        if isinstance(origin,dict):parent=parent or origin.get('source_sha256') or origin.get('sha256')
        note=obj.get('method') or obj.get('summary') or obj.get('attribution') or ''
        if child and isinstance(child,str) and path in opdocs:methods[key(child)]=dict(note=note,evidence=path)
        edge(parent,child,'explicit_parent',path,note)
    replay_path=Path('development/experiments/lineage/20261117/replayed-edges-v7.json')
    if replay_path.exists():
        for item in read(replay_path)['edges']:
            parent,child=item['parent_sha256'],item['source_sha256']
            if parent in keys and child in keys:
                edge(parent,child,item['kind'],
                     dict(index=str(replay_path),log=item['log'],log_sha256=item['log_sha256'],steps=item['steps']))
    for name in ('literal-proposal-edges.json','scatter-repair-edge.json','source-prefix-edges.json','wrapped-source-edges.json'):
        literal_path=Path('development/experiments/lineage/20261117')/name
        if not literal_path.exists():continue
        for item in read(literal_path)['edges']:
            if item['parent_sha256'] in keys and item['source_sha256'] in keys:
                if item['kind']=='exact_source_prefix':
                    parent=sources[item['parent_sha256']].read_bytes()
                    child=sources[item['source_sha256']].read_bytes()
                    assert digest(parent)==item['parent_sha256'] and digest(child)==item['source_sha256']
                    assert child.startswith(parent.rstrip()) and len(child)>len(parent.rstrip())
                    assert child[len(parent.rstrip()):len(parent.rstrip())+1] in (b'\r',b'\n')
                edge(item['parent_sha256'],item['source_sha256'],item['kind'],item)
    history=row['historical_ledger_entries'];groups=collections.defaultdict(list)
    for item in history:groups[item['ledger_sha256']].append(item)
    for lid,items in groups.items():
        byround={x['round']:x for x in items}
        for item in items:
            p=(item.get('plan',{}).get('source') or {}).get('parent_round_num')
            if p in byround:edge(byround[p].get('source_sha256'),item.get('source_sha256'),'ledger_parent',lid,
                                 item.get('plan',{}).get('strategy',''))
    matrix=read(Path('development/reports/trace-attribution-matrix-20261116.json'))
    ledger_files=next(x for x in matrix['rows'] if x['operator']==op)['ledger_files']
    ledger_paths={x['sha256']:x['path'] for x in ledger_files}
    for path in opdocs:
        if Path(path).name != 'plan.json':continue
        plan=docs[path]['data']
        if not isinstance(plan,dict):continue
        records=plan.get('records',[])
        if not isinstance(records,list):continue
        seed=next((r.get('source_sha256') for r in records
                   if isinstance(r,dict) and r.get('operator')==op and r.get('seed')),None)
        if not seed:continue
        for lid,items in groups.items():
            if campaign(ledger_paths[lid]) != campaign(path):continue
            # A fixed batch seed precedes its first generated ledger version.
            # Subsequent rounds must still use their recorded parent_round_num.
            first=min(items,key=lambda x:x['round'])
            edge(seed,first.get('source_sha256'),'batch_plan_to_first_round',path)
    for m in row['measurements']:
        sha=m.get('sources',[{}])[0].get('sha256') if m.get('sources') else None
        for path in links.get(m['result']['path'],{}).get('documents',[]):
            if Path(path).name!='input-provenance.json':continue
            seed=docs[path]['data'].get('source_sha256') or docs[path]['data'].get('seed_sha256')
            edge(seed,sha,'batch_input_to_selected',path,'Batch input to selected output; includes multiple internal edits')
            # Link the batch input to its first ledger round when the selected source identifies the ledger.
            for lid,items in groups.items():
                if any(x.get('source_sha256')==sha for x in items):
                    first=min(items,key=lambda x:x['round'])
                    edge(seed,first.get('source_sha256'),'batch_input_to_first_round',path)
    adjacency=collections.defaultdict(list)
    for e in edges:adjacency[e['parent']].append(e)
    @lru_cache(maxsize=None)
    def path_between(a,b):
        a,b=key(a),key(b)
        if a==b:return []
        queue=collections.deque([(a,[])]);seen={a}
        while queue:
            node,path=queue.popleft()
            for e in adjacency[node]:
                if e['child']==b:return path+[e]
                if e['child'] not in seen:seen.add(e['child']);queue.append((e['child'],path+[e]))
        return None
    events=[]
    for m in row['measurements']:
        if m['role']=='reference' or not m.get('sources'):continue
        if m['constraint_status']=='known_noncompliant':continue
        # Historic explicitly Ascend C candidates remain in the evidence table, outside the strict line.
        if re.search(r'/ascendc[-/]',m['result']['path'],re.I):continue
        if order(m)[0]>'20261109':continue
        sha=m['sources'][0]['sha256'];ref,refnote=reference(m,bundles,linkroot)
        is_delivery=sha==delivery['sha256'] and m['measurement']['geo_mean']==delivery['cann_score']
        if is_delivery:ref,refnote='CANN','Current retained delivery contract'
        ev=dict(id=digest(m['result']['path'].encode())[:16],source_sha256=sha,source_key=key(sha),
                score=m['measurement']['geo_mean'] if m['measurement']['status']=='PASSED' else None,
                status=m['measurement']['status'],reference=ref,reference_evidence=refnote,
                record=m,is_delivery=is_delivery,is_baseline=m['result']['path']==baseline['result']['path'],
                stage=campaign(m['result']['path']),sort=list(order(m)),method=methods.get(key(sha)))
        events.append(ev)
    matrix=read(Path('development/reports/trace-attribution-matrix-20261116.json'))
    ledger_files=next(x for x in matrix['rows'] if x['operator']==op)['ledger_files']
    for h in history:
        ledger=next(x for x in ledger_files if x['sha256']==h['ledger_sha256'])
        path=ledger['path'];stage=campaign(path);sha=h.get('source_sha256')
        if not sha:continue
        date=(h['evaluation'].get('evaluated_at') or '').replace('-','')[:8]
        # The legacy ledger uses a PyTorch contract; keep its original measurement,
        # not the later independently remeasured score of the same source.
        matching=[e for e in events if e['source_sha256']==sha]
        same_stage=[e for e in matching if e['stage']==stage]
        ref=same_stage[0]['reference'] if same_stage else 'PyTorch'
        rec=dict(result=dict(path=path,sha256=h['ledger_sha256']),ledger_round=h['round'],ledger=h,
                 sources=[dict(sha256=sha)],measurement=h['evaluation'])
        sort=list(order(rec));sort[0]=date or sort[0];sort[3]=1
        sort[4]='ledger-'+str(h['round']).zfill(4)
        ev=dict(id=h['ledger_sha256'][:10]+'-'+str(h['round']),source_sha256=sha,source_key=key(sha),
                score=h['evaluation'].get('geo_mean') if h['evaluation'].get('status')=='PASSED' else None,
                status=h['evaluation'].get('status'),reference=ref,reference_evidence='Original optimization ledger; initial PyTorch contract',
                record=rec,is_delivery=False,is_baseline=False,stage=stage,sort=sort,
                method=dict(note=h.get('plan',{}).get('strategy',''),evidence=path))
        events.append(ev)
    baseline_ev=next(e for e in events if e['is_baseline']);final=next(e for e in events if e['is_delivery'])
    stamps={}
    for h in history:
        if h.get('source_sha256') and h['evaluation'].get('evaluated_at'):
            k=key(h['source_sha256']);stamp=h['evaluation']['evaluated_at']
            stamps[k]=min(stamps.get(k,stamp),stamp)
    for e in events:
        stamp=stamps.get(e['source_key'])
        if stamp and e['sort'][0]==stamp[:10].replace('-',''):
            e['sort'][3]=1;e['sort'][4]=stamp
    # Same source under two references is a factual bridge, not a performance gain.
    bysource=collections.defaultdict(list)
    for e in events:
        if e['score'] is not None:bysource[e['source_sha256']].append(e)
    bridges=[]
    for sha,es in bysource.items():
        pt=sorted([e for e in es if e['reference']=='PyTorch'],key=lambda e:e['sort'])
        cn=sorted([e for e in es if e['reference']=='CANN'],key=lambda e:e['sort'])
        if pt and cn:bridges.append(dict(source_sha256=sha,pytorch=pt[0],cann=cn[0],delivery_source=sha==delivery['sha256']))
    # Preserve the main historical batch selected earlier, now add all subsequent source stages.
    seen=set();unique=[]
    for e in sorted(events,key=lambda e:e['sort']):
        sig=(e['source_key'],e['reference'])
        if e['is_baseline'] or e['is_delivery']:continue
        if sig in seen and e['record'].get('ledger_round')!=1:continue
        seen.add(sig);unique.append(e)
    # Avoid reintroducing an older source after final delivery; retain exact final measurement explicitly.
    unique=[e for e in unique if not (e['source_key']==final['source_key'] and e['reference']=='CANN')]
    ordered=[baseline_ev]+[e for e in unique if e['source_key']!=baseline_ev['source_key']]+[final]
    # Reorder explicit parent stages within the same dated phase; never assert lineage from order alone.
    for _ in range(len(ordered)):
        moved=False
        for i in range(1,len(ordered)-1):
            for j in range(i+1,len(ordered)-1):
                a,b=ordered[i],ordered[j]
                if a['reference']!=b['reference'] or a['sort'][0]!=b['sort'][0]:continue
                if path_between(b['source_sha256'],a['source_sha256']) and path_between(a['source_sha256'],b['source_sha256']) is None:
                    ordered.insert(i,ordered.pop(j));moved=True;break
            if moved:break
        if not moved:break
    connections=[]
    for a,b in zip(ordered,ordered[1:]):
        path=path_between(a['source_sha256'],b['source_sha256'])
        kind='same_source' if a['source_key']==b['source_key'] else ('recorded_lineage' if path is not None else 'unresolved')
        connections.append(dict(start=a['id'],end=b['id'],kind=kind,reference_change=a['reference']!=b['reference'],proof=path))
    whole=path_between(baseline_ev['source_sha256'],final['source_sha256'])
    # The primary view follows a proved ancestry route when available. Exploration
    # branches remain in full_history, rather than interrupting the delivery spine.
    full_history=ordered
    if whole is not None:
        route=[key(baseline_ev['source_sha256'])]+[e['child'] for e in whole]
        selected=[baseline_ev]
        for sourcekey in route[1:]:
            refs=set()
            for e in sorted((x for x in ordered if x['source_key']==sourcekey),key=lambda x:x['sort']):
                if e['is_delivery']:continue
                if sourcekey==final['source_key'] and e['reference']=='CANN':continue
                if e['reference'] in refs:continue
                refs.add(e['reference']);selected.append(e)
        selected.append(final)
        ordered=selected
    # Prefer retained parent branches: remove an offshoot only when a later node
    # explicitly descends from an earlier displayed version and not the offshoot.
    changed=True
    while changed:
        changed=False
        for i in range(1,len(ordered)-1):
            a,b,c=ordered[i-1:i+2]
            ab=path_between(a['source_sha256'],b['source_sha256']);ac=path_between(a['source_sha256'],c['source_sha256']);bc=path_between(b['source_sha256'],c['source_sha256'])
            if ab and ac is not None and bc is None and not b['is_baseline'] and not b['is_delivery']:
                ordered.pop(i);changed=True;break
    connections=[]
    for a,b in zip(ordered,ordered[1:]):
        path=path_between(a['source_sha256'],b['source_sha256'])
        kind='same_source' if a['source_key']==b['source_key'] else ('recorded_lineage' if path is not None else 'unresolved')
        connections.append(dict(start=a['id'],end=b['id'],kind=kind,reference_change=a['reference']!=b['reference'],proof=path))
    return dict(operator=op,delivery=delivery,baseline_correction=correction,events=events,display_events=ordered,full_history=full_history,connections=connections,
                edges=edges,bridges=bridges,full_lineage_verified=whole is not None,full_lineage_proof=whole,
                counts=dict(display_nodes=len(ordered),same_source_connections=sum(c['kind']=='same_source' for c in connections),
                            recorded_connections=sum(c['kind']=='recorded_lineage' for c in connections),
                            unresolved_connections=sum(c['kind']=='unresolved' for c in connections)))


def method_label(event, known):
    if event.get('is_baseline') or event.get('attribution_status')=='version_selection_or_retest':return None
    if event.get('attribution'):
        record=event['attribution']
        return dict(method=record['method']+('*' if record['confidence']=='inferred' else ''),
                    color=record['primary'],location=record['location'],attribution=record)
    if event.get('recovery_method'):return event['recovery_method']
    if event['source_sha256'] in known:return known[event['source_sha256']]
    path=event['record']['result']['path'].lower();note=(event.get('method') or {}).get('note','')
    choices=[('constexpr-layout','Compile-time layout specialization'),('tiled-2d-launch','2-D contiguous tiling'),
             ('integer-metadata','Simplify host metadata math'),('fp32-q-accumulation','Repair Q accumulation precision'),
             ('cpp-host','Simplify C++ host path'),('codex-disjoint','Repair contiguous-tile boundaries'),
             ('transfer-log10','Transfer multi-tile method'),('full-tile-stream','Full-tile streaming'),
             ('half-tiles','Adjust low-precision tiling'),('launch-once','Reduce launch-wrapper cost'),
             ('restored-v1','Verify strict Triton paths')]
    for token,label in choices:
        if token in path:
            color='probe' if 'host-profile-guided' in note.lower() else 'ir'
            return dict(method=label,color=color,location='host' if 'Host' in label or 'Launch' in label else 'mixed')
    return None


def svg(row,known,all_history=False):
    from .selection import remeasurement_indices, contract_connections
    original=row['full_history'] if all_history else row['display_events']
    hidden=remeasurement_indices(original)
    es=[event for i,event in enumerate(original) if i not in hidden]
    row=dict(row,connections=contract_connections(original,row['connections'],es))
    n=len(es);w=max(1600,320+245*(n-1));height=1060
    left,right,top,bottom=180,w-125,355,735
    xs=[left+(right-left)*i/max(n-1,1) for i in range(n)]
    valid=[e['score'] for e in es if e['score'] is not None and e['score']>0]
    hi=max(valid)*1.18;low=0;log=max(valid)/min(valid)>100
    if log:
        low=math.floor(math.log10(min(valid)));high=math.ceil(math.log10(max(valid)*1.15))
        y=lambda v:bottom-(math.log10(v)-low)/(high-low)*(bottom-top)
        ticks=[10**i for i in range(low,high+1)]
    else:
        hi=max(.5,math.ceil(hi*10)/10);y=lambda v:bottom-v/hi*(bottom-top);ticks=[hi*i/4 for i in range(1,5)]
    ys=[y(e['score']) if e['score'] is not None and e['score']>0 else 785 for e in es]
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{height}" viewBox="0 0 {w} {height}">',
           f'<rect width="{w}" height="{height}" fill="{BG}"/>',
           '<defs>'+''.join(f'<marker id="method-{cat}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="{color}"/></marker>' for cat,color in COLORS.items())+'<marker id="arr" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#596170"/></marker></defs>',
           '<g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">',
           text(52,62,row['operator'],min(42, (w-650)/(len(row['operator'])*.58))),
           text(52,112,('Delivery ' if row.get('progress_view') else '')+f"{row['delivery']['cann_score']:.3f}× CANN",30,GREEN)]
    used=['paired','knowledge','ir','sim']
    names={'paired':'Profiling','knowledge':'Knowledge / experience','sim':'Simulator','ir':'Source / IR analysis'}
    for k,cat in enumerate(used):
        x=w-520+(k%2)*255;ly=55+(k//2)*40
        parts += [f'<line x1="{x}" y1="{ly}" x2="{x+23}" y2="{ly}" stroke="{COLORS[cat]}" stroke-width="3"/>',text(x+32,ly+8,names[cat],23,COLORS[cat])]
    for k,label in enumerate(['Device kernel','Host','Mixed','Start']):
        x=w-520+(k%2)*255;ly=145+(k//2)*40
        if k==0:mark=f'<circle cx="{x+10}" cy="{ly}" r="8" fill="{GREEN}"/>'
        elif k==1:mark=f'<rect x="{x+2}" y="{ly-8}" width="16" height="16" fill="{GREEN}"/>'
        elif k==2:mark=f'<path d="M{x+10} {ly-10}L{x+20} {ly}L{x+10} {ly+10}L{x} {ly}Z" fill="{GREEN}"/>'
        else:mark=f'<circle cx="{x+10}" cy="{ly}" r="8" fill="#92908a"/>'
        parts += [mark,text(x+32,ly+8,label,23)]
    # Reference spans label their own denominators, without changing the recorded values.
    spans=[];start=0
    for i in range(1,n+1):
        if i==n or es[i]['reference']!=es[start]['reference']:
            spans.append((start,i-1,es[start]['reference']));start=i
    if not row.get('normalized_view'):
        for a,b,ref in spans:
            x1=xs[a]-48;x2=xs[b]+48
            parts += [f'<path d="M{x1} 194H{x2}" stroke="#c9c1ae" stroke-width="3"/>',text((x1+x2)/2,230,ref,25,anchor='middle')]
    parts.append(text(52,320,'Speedup'+('(log scale)' if log else ''),27))
    for tick in ticks:
        label=f'{tick:g}×' if log else f'{tick:.2f}×'
        parts += [f'<line x1="150" y1="{y(tick):.2f}" x2="{w-65}" y2="{y(tick):.2f}" stroke="#ded7c9"/>',text(125,y(tick)+9,label,25,anchor='end')]
    c_by_pair={(c['start'],c['end']):c for c in row['connections']}
    transfer=row.get('method_transfer') if not all_history and not row.get('progress_view') else None
    transfer_targets=transfer.get('display_targets',[transfer['display_target_sha256']]) if transfer else []
    for i in range(n-1):
        a,b=es[i:i+2];conn=c_by_pair.get((a['id'],b['id']))
        same=a['source_key']==b['source_key']
        confirmed=same or (conn and conn['kind'] in ('recorded_lineage','recorded_recovery','progress_summary'))
        if row.get('progress_view') and conn and conn['kind']=='documented_method_switch':
            # The presentation curve tracks measured performance progress.
            # The recorded source branch remains a separate relation in details.
            detail=html.escape(json.dumps(conn,ensure_ascii=False))
            parts.append(f'<path class="connection" d="M{xs[i]:.2f} {ys[i]:.2f} L{xs[i+1]:.2f} {ys[i+1]:.2f}" stroke="{GREEN}" stroke-width="3.5" fill="none"><title>{detail}</title></path>')
            continue
        refchange=a['reference']!=b['reference']
        color=GREEN if confirmed and not refchange else '#596170'
        dash='' if confirmed and not refchange else 'stroke-dasharray="8 8"'
        # A documented imported implementation starts a new ancestry branch.
        # Do not draw a false parent edge from the preceding same-name version.
        if b['source_sha256'] in transfer_targets and not confirmed:
            # Chronological performance progression is drawn explicitly as a
            # scheme switch, never added to the source-ancestry edge graph.
            parts.append(f'<path class="method-switch" d="M{xs[i]:.2f} {ys[i]:.2f} L{xs[i+1]:.2f} {ys[i+1]:.2f}" stroke="#596170" stroke-width="2" fill="none"/>')
            parts.append(text((xs[i]+xs[i+1])/2,(ys[i]+ys[i+1])/2-24,'Method switch',23,anchor='middle'))
            continue
        parts.append(f'<path class="connection" d="M{xs[i]:.2f} {ys[i]:.2f} L{xs[i+1]:.2f} {ys[i+1]:.2f}" stroke="{color}" stroke-width="3.5" fill="none" {dash}/>')
        if refchange:parts.append(text((xs[i]+xs[i+1])/2,285,'Reference change',23,anchor='middle'))
    if transfer:
        if transfer.get('attribution'):
            from .attribution import tooltip
            parts.append('<g><title>'+html.escape(tooltip(transfer['attribution']))+'</title>')
        target=next(i for i,e in enumerate(es) if e['source_sha256']==transfer_targets[-1])
        tx=xs[target];bx=max(410,min(w-410,tx-180));by=270
        parts += [f'<rect class="donor" x="{bx-300}" y="{by-32}" width="600" height="62" rx="12" fill="{BG}" stroke="{COLORS["knowledge"]}" stroke-width="2"/>',
                  text(bx,by+8,transfer['donor']+' -> method transfer',27,COLORS['knowledge'],anchor='middle')]
        for digest in transfer_targets:
            target=next(i for i,e in enumerate(es) if e['source_sha256']==digest);tx=xs[target]
            parts.append(f'<path class="donor-connection" d="M{bx} {by+30} Q{bx} {ys[target]} {tx-18} {ys[target]}" stroke="{COLORS["knowledge"]}" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>')
        if transfer.get('attribution'):parts.append('</g>')
    annotations=[]
    for i,e in enumerate(es):
        method=method_label(e,known)
        # Validation and version selection remain in the evidence, rather than
        # being presented as optimization techniques in the figure.
        administrative = method and any(token in method['method'] for token in
                                        ('Verify strict', 'Restore historically qualified version'))
        if method and not administrative:annotations.append((i,method))
    # Every mainline modification receives a visible label; no top-k truncation.
    for k,(i,m) in enumerate(annotations):
        if m.get('attribution'):
            from .attribution import tooltip
            parts.append('<g class="attribution"><title>'+html.escape(tooltip(m['attribution']))+'</title>')
        width=min(320,(right-left)/max(n-1,1)-28)
        tx=max(width/2+30,min(w-width/2-30,xs[i]))
        # Place the method opposite the numeric score, close to its version.
        ty=ys[i]-110 if i%2 else ys[i]+85
        color=COLORS[m['color']]
        lines=[];line='';size=0
        for char in re.findall(r'[A-Za-z0-9_+]+|.',m['method']):
            cw=sum(28 if ord(c)>127 else 16 for c in char)
            if size+cw>width and line and char!='*':lines.append(line);line='';size=0
            line+=char;size+=cw
        if line:lines.append(line)
        parts.extend(text(tx,ty+j*33,line,28,color,anchor='middle') for j,line in enumerate(lines))
        start=ys[i]-45 if i%2 else ys[i]+50
        end=ys[i]-18 if i%2 else ys[i]+18
        parts.append(f'<path d="M{xs[i]:.2f} {start:.2f}L{xs[i]:.2f} {end:.2f}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#method-{m["color"]})"/>')
        if m.get('attribution'):parts.append('</g>')
    for i,e in enumerate(es):
        x,yy=xs[i],ys[i];m=method_label(e,known);loc=e.get('change_location',{}).get('location') or (m['location'] if m else None)
        parts.append(f'<g class="node" data-id="{e["id"]}"><title>{html.escape(e["source_sha256"]+" | "+e["stage"]+" | "+e["reference"])}</title>')
        if e['score'] is None:
            parts.append(f'<path d="M{x-9} {yy-9}L{x+9} {yy+9}M{x-9} {yy+9}L{x+9} {yy-9}" stroke="#b84c42" stroke-width="4"/>')
        elif loc=='host':parts.append(f'<rect x="{x-9}" y="{yy-9}" width="18" height="18" fill="{GREEN}" stroke="{BG}" stroke-width="2"/>')
        elif loc=='mixed':parts.append(f'<path d="M{x} {yy-12}L{x+12} {yy}L{x} {yy+12}L{x-12} {yy}Z" fill="{GREEN}" stroke="{BG}" stroke-width="2"/>')
        else:parts.append(f'<circle cx="{x}" cy="{yy}" r="9" fill="{GREEN if loc=="kernel" else "#92908a"}" stroke="{BG}" stroke-width="2"/>')
        label=(f"{e['score']:.4f}×" if e['score']<.01 else f"{e['score']:.3f}×") if e['score'] is not None else 'Failed'
        if e.get('score_kind')=='rebased':label='≈'+label
        parts.append(text(x,yy+38 if i%2 else yy-23,label,27,anchor='middle'))
        parts.append(text(x,1005,'FlagGems' if e['is_baseline'] else (f'V{i} | delivery ' if e['is_delivery'] else f'V{i}'),25,anchor='middle'))
        parts.append('</g>')
    parts+=['</g></svg>'];return '\n'.join(parts)


def render_rows(rows,out):
    catalog=read(Path('development/reports/operator-traces-26-20261116/catalog.json'))
    known={}
    for r in catalog['operators']:
        for a in r['view']['annotations']:
            p=r['points'][a['point']-1]
            location=next((x['location'] for x in r['view'].get('nodes',[]) if x['point']==a['point']),'mixed')
            known[p.get('source_sha256')]=dict(method=a['method'],color=a['color'],location=location)
    cards=[];report=['# 26-operator source ancestry','','Each chart starts at native FlagGems submission and ends at current delivery. Solid links mean verified same-source/static equivalence or explicit parents; dashed links retain unverified relations. Mark reference changes separately from gains.',
        '', 'Indices count displayed versions, not every model round. Prefer proven delivery ancestry; retain other attempts in full history.',
        '', '| Operator | Main nodes | Root-to-delivery ancestry | Same-source dual reference | Delivery dual reference |','| --- | ---: | --- | --- | --- |']
    css="body{margin:30px;background:#fff7e8;color:#1b2434;font:20px/1.65 'Microsoft YaHei',sans-serif}h1{font-size:34px}a{color:#2368b1}.viewport{overflow:auto;border:1px solid #ded7c9}svg{display:block;max-width:none}table{border-collapse:collapse;width:100%;font-size:17px}td,th{border-bottom:1px solid #d8d5c8;padding:10px;text-align:left;vertical-align:top}code{overflow-wrap:anywhere}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}.cards img{width:100%}.card{border:1px solid #ded7c9;padding:12px}"
    for row in rows:
        op=row['operator'];dest=out/op;dest.mkdir(exist_ok=True)
        (dest/'mainline.svg').write_text(svg(row,known),encoding='utf8')
        (dest/'history.svg').write_text(svg(row,known,True),encoding='utf8')
        save(dest/'trace.json',row)
        table=[]
        for i,e in enumerate(row['display_events']):
            m=method_label(e,known);label=m['method'] if m else ((e.get('method')or{}).get('note') or 'Method record incomplete')
            c=row['connections'][i-1] if i else None
            relation='Original start' if not c else {'same_source':'Same source / static equivalence','recorded_lineage':'Explicit parent connection','unresolved':'Connection unverified'}[c['kind']]
            if c and c['reference_change']:relation+='; reference change'
            table.append('<tr><td>'+str(i)+'</td><td>'+html.escape(e['stage'])+'</td><td>'+('Failed' if e['score'] is None else f"{e['score']:.6f}×")+' '+e['reference']+'</td><td><code>'+e['source_sha256'][:12]+'</code><br>'+relation+'</td><td>'+html.escape(label)+'</td></tr>')
        status='Root-to-delivery ancestry located' if row['full_lineage_verified'] else 'Some segments connected; dashed links remain unverified'
        page=f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{op} source ancestry</title><style>{css}</style><p><a href="../index.html">26-operator overview</a> | <a href="mainline.png">PNG</a> | <a href="mainline.svg">SVG</a> | <a href="trace.json">Full evidence</a></p><h1>{op}</h1><p>{status}. Scroll for all nodes; reference changes are separate from code optimization.</p><div class="viewport">'+svg(row,known)+'</div><p>Colors retain verified guidance. Hollow nodes have unclassified change locations; circle/square/diamond denote kernel/host/mixed.</p><details><summary>All history and branches outside delivery ancestry</summary><div class="viewport">'+svg(row,known,True)+'</div></details><table><tr><th>Node</th><th>Batch</th><th>Score/reference</th><th>Source/ancestry</th><th>Method record</th></tr>'+''.join(table)+'</table></html>'
        (dest/'index.html').write_text(page,encoding='utf8')
        cards.append(f'<div class="card"><h2><a href="{op}/index.html">{op}</a></h2><a href="{op}/index.html"><img src="{op}/mainline.svg"></a><p>Delivery {row["delivery"]["cann_score"]:.3f}× CANN · {status}</p></div>')
        report.append(f'| [{op}]({op}/index.html) | {len(row["display_events"])} | '+('Located' if row['full_lineage_verified'] else 'Partially connected')+' | yes | '+('Yes' if any(b['delivery_source'] for b in row['bridges']) else 'Latest delivery: CANN only')+' |')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>26-operator source ancestry</title><style>'+css+'</style><h1>26 operators: FlagGems to delivery</h1><p>Solid: verified ancestry. Dashed: unverified links/reference switches, marked separately.</p><p><a href="README.md">Audit findings</a> | <a href="reference-bridges.json">Same-source reference evidence</a></p><div class="cards">'+''.join(cards)+'</div></html>',encoding='utf8')
    complete=[r['operator'] for r in rows if r['full_lineage_verified']]
    missing_bridge=[r['operator'] for r in rows if not any(b['delivery_source'] for b in r['bridges'])]
    report+=['','## Reference-switch effects','',
             'All 26 have a same-source PyTorch/CANN pair; 22 use current delivery source. Four latest deliveries currently have only CANN results: '+ '、'.join('`'+x+'`' for x in missing_bridge)+'. Earlier versions of those four still have dual-reference records.',
             '', '| Same source | PyTorch reference | CANN reference |','| --- | ---: | ---: |']
    for op in ('upsample_bilinear2d','transform_bias_rescale_qkv','log2_','_conj_physical','batch_norm_with_update_functional'):
        r=next(x for x in rows if x['operator']==op);b=next(x for x in r['bridges'] if x['delivery_source'])
        report.append(f"| `{op}` | {b['pytorch']['score']:.6f}× | {b['cann']['score']:.6f}× |")
    report+=['','These are different-reference measurements of the same code, including measurement variation, not code optimization gains. Source ancestry may be valid while performance interpretation remains segmented. Preserve historical PyTorch CPU-fallback scope, especially for large _conj_physical ratios.',
             '', '## Completeness','',
             'Native starts/delivery endpoints were found for all 26. Explicit parents, batch input/output, and static equivalence establish six main root-to-delivery histories: '+ '、'.join('`'+x+'`' for x in complete)+'. This does not imply a separate score for each internal edit.',
             '', 'The other 20 have connectable segments but at least one unresolved ancestry link, marked dashed and retained in JSON. Reference changes do not invalidate verified same-source connections.',
             '', 'Among 26 fixed-batch native starts, nine passed correctness below 0.8, 14 failed full acceptance, and three were already at least 0.8 under historical references. Do not assign zero to failures or lower faster baselines.',
             '', 'Main charts fold same-source/reference repeats while preserving first ledger rounds/delivery. Prefer proven ancestors. Keep other attempts expandable and known Ascend C mixed branches separate from strict ancestry. Unknown-compliance nodes are not automatically strict.',
             '', 'Submitted parent records: `development/experiments/lineage/20261117/links-v2/`. Recovered 426 provenance documents, 210 contract identities, and 241 SHA-matched sources. Nineteen referenced source files were not located with original SHA; links.json records gaps. No new device experiments.',
             '', '## Next steps','',
             'Recover 20 missing parent/invocation links without rerunning all 26. Unified reevaluation is separate work when required reference timings/common scales are missing; do not present remeasurement as original history.',
             '', '```powershell','python -m development.tools.traces.stitched --output development/reports/operator-traces-stitched-20261117 --png','```','']
    (out/'README.md').write_text('\n'.join(report)+'\n',encoding='utf8')


def export_check(out):
    from playwright.sync_api import sync_playwright
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page()
        for f in out.glob('*/mainline.svg'):
            page.goto(f.as_uri());dims=page.locator('svg').evaluate('s=>({width:s.width.baseVal.value,height:s.height.baseVal.value})')
            page.set_viewport_size({k:int(v) for k,v in dims.items()});page.screenshot(path=str(f.with_suffix('.png')))
            issues=page.locator('svg').evaluate('s=>[...s.querySelectorAll("text")].filter(e=>{let b=e.getBBox();return b.x<0||b.x+b.width>s.width.baseVal.value+1||b.y<0||b.y+b.height>s.height.baseVal.value+1}).map(e=>e.textContent)')
            checks.append(dict(operator=f.parent.name,overflow=issues))
        browser.close()
    save(out/'presentation-check.json',checks)
    assert len(checks)==26 and all(not x['overflow'] for x in checks),checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--png',action='store_true')
    a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    evidence=root/'development/experiments/lineage/20261117/source-evidence';linkroot=root/'development/experiments/lineage/20261117/links-v9'
    data=read(root/'development/reports/native-to-delivery-26-20261117/inventory.json');links=read(linkroot/'links.json')
    rows=[build(r,evidence,linkroot,links) for r in data['operators']]
    save(out/'graph.json',dict(operators=rows))
    save(out/'reference-bridges.json',dict(operators=[dict(operator=r['operator'],bridges=r['bridges']) for r in rows]))
    render_rows(rows,out)
    if a.png:export_check(out)
    print(json.dumps({'operators':len(rows),'same_source_reference_bridge':sum(bool(r['bridges']) for r in rows),
        'delivery_source_reference_bridge':sum(any(b['delivery_source'] for b in r['bridges']) for r in rows),
        'full_lineage_verified':sum(r['full_lineage_verified'] for r in rows),
        'nodes':sum(r['counts']['display_nodes'] for r in rows)},ensure_ascii=False))


if __name__=='__main__':main()
