"""Collect existing nearby evaluation provenance/contracts read-only, without experiments."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.inventory.read_bytes());a.output.mkdir(exist_ok=False,parents=True)
    store=a.output/'artifacts';store.mkdir()
    names={'proposal.json','review.json','input-provenance.json','source-provenance.json','summary.json',
           'agent-context-loaded.json','selected.json','manifest.json','provenance.json','origin.json',
           'extra-checks.json','supplementary.json','cann_supplementary.json','gate.json',
           'source-review.json','assessment.json','performance-feedback.json',
           'review-exclusions.json','baseline-verified.json','final-delivery.json',
           'starter-provenance.json','plan.json','spec.json','extra-cann.json',
           'extra-pytorch.json','runtime-audit.json','cold-start-control-check.json',
           'backend-verification.json','selected-runtime.json'}
    docs={};bundles={};records=[];errors=[];sources=[];unresolved=[]
    for item in data['records']:
        result=Path(item['result']['path']);ancestors=[];base=result.parent
        # Include the operator directory, but not campaign-global unrelated metadata.
        for _ in range(7):
            ancestors.append(base)
            if base.name in (item['operator'],'conj_physical'):break
            base=base.parent
        # Batch-level plans and exclusions bind seeds and rejected sources;
        # read only the explicit metadata whitelist, not arbitrary batch files.
        for parent in result.parents:
            if parent.parent.name == 'campaigns' and parent not in ancestors:
                ancestors.append(parent)
                break
        record={'result_path':str(result),'operator':item['operator'],'documents':[],'bundles':[]}
        for base in ancestors:
            for name in names:
                f=base/name
                if not f.is_file():continue
                try:
                    if str(f) not in docs:
                        raw=f.read_bytes();obj=json.loads(raw);h=digest(raw);(store/(h+'.json')).write_bytes(raw)
                        docs[str(f)]={'path':str(f),'sha256':h,'export':'artifacts/'+h+'.json','data':obj}
                    record['documents'].append(str(f))
                except Exception as e:errors.append({'path':str(f),'error':str(e)})
            f=base/'operator-bundle.tar'
            if not f.is_file():continue
            try:
                key=str(f)
                if key not in bundles:
                    raw=f.read_bytes();members=[]
                    with tarfile.open(str(f)) as tar:
                        for member in tar.getmembers():
                            if not member.isfile():continue
                            short=Path(member.name).name
                            if short not in ('oracle.py','definition.json','timing.jsonl','correctness.jsonl'):continue
                            body=tar.extractfile(member).read();h=digest(body);dest=store/(h+Path(short).suffix)
                            dest.write_bytes(body)
                            members.append({'member':member.name,'sha256':h,'export':dest.relative_to(a.output).as_posix()})
                    bundles[key]={'path':key,'sha256':digest(raw),'bundle_id':'sha256:'+digest(raw),'members':members}
                record['bundles'].append(key)
            except Exception as e:errors.append({'path':str(f),'error':str(e)})
        records.append(record)
    # Unmeasured intermediate proposals can bridge two measured versions.
    campaign_roots=set()
    for item in data['records']:
        for parent in Path(item['result']['path']).parents:
            if parent.parent.name=='campaigns':campaign_roots.add(parent);break
    for base in campaign_roots:
        for f in base.glob('*provenance*.json'):
            if str(f) in docs:continue
            try:
                raw=f.read_bytes();obj=json.loads(raw);h=digest(raw)
                (store/(h+'.json')).write_bytes(raw)
                docs[str(f)]={'path':str(f),'sha256':h,'export':'artifacts/'+h+'.json','data':obj}
                for record in records:
                    if str(base) in record['result_path']:record['documents'].append(str(f))
            except Exception as e:errors.append({'path':str(f),'error':str(e)})
        for filename in ('spec.json','proposal.json','input-provenance.json','source-provenance.json','review.json'):
            for f in base.rglob(filename):
                if str(f) in docs:continue
                try:
                    raw=f.read_bytes();obj=json.loads(raw)
                    if not isinstance(obj,dict):continue
                    operator=obj.get('operator')
                    if not operator:
                        operator=next((op for op in {i['operator'] for i in data['records']}
                                       if op in f.parts),None)
                    if not operator:continue
                    h=digest(raw);(store/(h+'.json')).write_bytes(raw)
                    docs[str(f)]={'path':str(f),'sha256':h,'export':'artifacts/'+h+'.json','data':obj}
                    for record in records:
                        if record['operator']==operator and str(base) in record['result_path']:
                            record['documents'].append(str(f))
                except Exception as e:errors.append({'path':str(f),'error':str(e)})
    # Resolve explicitly hash-bound parents/seeds without trusting mutable path names.
    archive='/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/'
    old='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'
    seen=set()
    for doc in docs.values():
        obj=doc['data']
        if isinstance(obj,dict) and isinstance(obj.get('source_sha256'),str):
            h=obj['source_sha256']
            for name in ('main.py','candidate.py'):
                source=Path(doc['path']).parent/name
                if source.is_file() and digest(source.read_bytes())==h and (str(source),h) not in seen:
                    seen.add((str(source),h));dest=store/(h+'.py');dest.write_bytes(source.read_bytes())
                    sources.append({'path':str(source),'sha256':h,'export':dest.relative_to(a.output).as_posix(),'declared_in':doc['path']})
        def visit(obj):
            if isinstance(obj,list):
                for item in obj:visit(item)
            if not isinstance(obj,dict):return
            for field,hashfield in [('source','source_sha256'),('seed','source_sha256'),
                                   ('seed','seed_sha256'),('origin','origin_sha256'),('parent','parent_sha256')]:
                path=obj.get(field);h=obj.get(hashfield)
                if not isinstance(path,str) or not isinstance(h,str) or len(h)!=64:continue
                if (path,h) in seen:continue
                seen.add((path,h))
                candidates=[Path(path),Path(doc['path']).parent/path,Path(archive)/path]
                if path.startswith(old):candidates.append(Path(archive+path[len(old):]))
                found=False
                for source in candidates:
                    if not str(source).startswith('/data/hanle/ascend-optimization/'):continue
                    if source.is_file() and source.suffix in ('.py','.cpp','.cc'):
                        raw=source.read_bytes()
                        if digest(raw)!=h:continue
                        dest=store/(h+source.suffix);dest.write_bytes(raw)
                        sources.append({'path':str(source),'sha256':h,'export':dest.relative_to(a.output).as_posix(),'declared_in':doc['path']})
                        found=True;break
                if not found:unresolved.append({'path':path,'sha256':h,'declared_in':doc['path']})
            for value in obj.values():
                if isinstance(value,(dict,list)):visit(value)
        visit(doc['data'])
    output={'schema':'hgkda.lineage-links/v1','records':records,'documents':list(docs.values()),'bundles':list(bundles.values()),'sources':sources,'unresolved_sources':unresolved,'errors':errors}
    (a.output/'links.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'documents':len(docs),'bundles':len(bundles),'sources':len(sources),'unresolved':len(unresolved),'errors':len(errors)}))


if __name__=='__main__':main()
