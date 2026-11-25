"""Search historical evaluations read-only; export source identities, artifact references, and baseline originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_bytes())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'artifacts').mkdir()
    (out/'sources').mkdir()
    ops = set(config['operators'])
    aliases = dict((op, op) for op in ops)
    aliases['conj_physical'] = '_conj_physical'
    rows, plans, errors = [], [], []
    excluded = {'.git', '.claude', '.codex', 'node_modules', '__pycache__', 'native-evidence',
                'profiles', 'method-package', 'method-history', 'trace', 'actual-ir',
                'controller-updates', 'file-updates', 'knowledge-updates', 'tool-replay',
                'tool-replay-v2', 'tool-versions', 'kg-runs', 'catalog-snapshots',
                'reference-audit-history', 'build-history', 'closing-delivery', 'closing-addendum'}

    def artifact(path, copy=False):
        raw = path.read_bytes()
        record = dict(path=str(path), sha256=sha(raw), bytes=len(raw))
        if copy:
            dest = out/'artifacts'/(record['sha256']+path.suffix)
            if not dest.exists():dest.write_bytes(raw)
            record['export'] = dest.relative_to(out).as_posix()
        return record

    for root_s in config['roots']:
        root = Path(root_s)
        for base_s, dirs, files in os.walk(str(root)):
            # Operator names such as cachemask_out are evidence, not cache directories.
            dirs[:] = [d for d in dirs if d not in excluded and d.lower() not in ('cache', '.cache', 'caches', 'venv', '.venv')
                       and not any(t in d.lower() for t in ('backup',))]
            base = Path(base_s)
            op = next((aliases[x] for x in reversed(base.parts) if x in aliases), None)
            if 'plan.json' in files and base.parent == root:
                try:
                    plan_path = base/'plan.json'
                    plan = json.loads(plan_path.read_bytes())
                    for rec in plan.get('records', []):
                        name = aliases.get(rec.get('operator'))
                        if name:
                            extra = dict(operator=name, campaign=base.name, record=rec, plan=artifact(plan_path, True))
                            seed = base/rec.get('seed', '__missing__')
                            if seed.is_file():extra['seed'] = artifact(seed, True)
                            leaf = base/rec.get('catalog_leaf', '__missing__')
                            extra['contract_files'] = [artifact(leaf/n, True) for n in ('definition.json','correctness.jsonl','timing.jsonl','oracle.py') if (leaf/n).is_file()]
                            for name2 in ('software-lock.json','baseline-verified.json','baseline-verified-final.json'):
                                if (base/name2).is_file():extra[name2] = artifact(base/name2, True)
                            plans.append(extra)
                except Exception as e:errors.append(dict(path=str(base/'plan.json'), error=str(e)))
            if not op:continue
            for name in files:
                if name != 'result.json' and not name.endswith('.result.json'):continue
                path = base/name
                try:
                    data = json.loads(path.read_bytes())
                    if not isinstance(data, dict) or 'geo_mean' not in data:continue
                    is_baseline = any(x == 'FlagGems' or x.lower().startswith('baseline') for x in base.parts)
                    row = dict(operator=op, baseline_path=is_baseline, result=artifact(path, is_baseline),
                               measurement={k:data.get(k) for k in ('status','geo_mean','num_passed','num_workloads','reference_source','device','server_backend','is_hack','timing_skipped')})
                    row['measurement']['timing_cases'] = sum(x.get('phase')=='timing' for x in data.get('per_workload', []))
                    row['measurement']['errors'] = [dict(uuid=x.get('uuid'),status=x.get('status'),log=str(x.get('log',''))[:2000]) for x in data.get('per_workload',[]) if x.get('status') not in ('PASSED',None)][:5]
                    request_path = path.with_name(name.replace('result.json','request.json'))
                    if request_path.exists():
                        req = json.loads(request_path.read_bytes())
                        row['request'] = artifact(request_path, is_baseline)
                        row['contract'] = {k:req.get(k) for k in ('api_version','binding','settings','reference_source','timing_scope')}
                        impl = req.get('implementation',{})
                        row['implementation'] = {k:impl.get(k) for k in ('name','definition','entrypoint','language')}
                        row['sources'] = []
                        for source in impl.get('sources',[]):
                            if not isinstance(source.get('content'), str):continue
                            raw = source['content'].encode('utf8')
                            ident = sha(raw)
                            ext = Path(source.get('path','')).suffix or '.txt'
                            dest = out/'sources'/(ident+ext)
                            if not dest.exists():dest.write_bytes(raw)
                            row['sources'].append(dict(name=source.get('path'),sha256=ident,export=dest.relative_to(out).as_posix()))
                    provenance = []
                    for f in files:
                        if f.endswith('.provenance.json') or f in ('review.json','source-provenance.json'):
                            provenance.append(artifact(base/f, True))
                    row['provenance'] = provenance
                    if is_baseline:
                        row['baseline_files'] = [artifact(base/f, True) for f in files if f in ('FlagGems.py','inspect.json','server-status.json','bundle-response.json')]
                    rows.append(row)
                except Exception as e:errors.append(dict(path=str(path),error=str(e)))
    result = dict(schema='hgkda.lineage-discovery/v1',scope=config,records=rows,plans=plans,errors=errors)
    (out/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(results=len(rows),baselines=sum(r['baseline_path'] for r in rows),plans=len(plans),errors=len(errors),output=str(out))))


if __name__=='__main__':main()
