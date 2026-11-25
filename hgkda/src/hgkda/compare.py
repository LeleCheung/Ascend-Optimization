"""Compare two OpProf captures for the same spec; activity ratios are evidence, not causal conclusions."""


import argparse


import csv


import hashlib


import json


import math


from pathlib import Path


import statistics
import re


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_side(directory, expected_spec_sha, side):
    meta = json.loads((directory/'collection.json').read_text(encoding='utf-8-sig'))
    if meta['status'] != 'completed' or meta['spec_sha256'] not in expected_spec_sha or meta['side'] != side:
        raise ValueError('Capture identity or completion status mismatch: '+str(directory))
    if meta.get('kernels'):
        children=[]
        for item in meta['kernels']:
            child_path=(directory/item['directory']).resolve()
            if not child_path.is_relative_to(directory.resolve()):raise ValueError('Kernel capture path escapes its directory')
            child=read_side(child_path,expected_spec_sha,side)
            if child['basic']['Op Name']!=item['symbol']:raise ValueError('Child kernel identity mismatch')
            if not isinstance(item['occurrences'],int) or item['occurrences']<1:raise ValueError('Invalid kernel occurrence count')
            children.append((child,item['occurrences']))
        devices={child['basic'].get('Device Id') for child,_ in children}
        if len(devices)!=1 or None in devices:raise ValueError('Multi-kernel capture contains different devices')
        tables={}
        for child,count in children:
            for key,group in child['tables'].items():
                target=tables.setdefault(key,dict(rows=0,fields={}))
                target['rows']+=group['rows']*count
                for field,stats in group['fields'].items():
                    value=target['fields'].setdefault(field,dict(valid=0,missing=0,min=None,max=None,_sum=0))
                    value['valid']+=stats['valid']*count;value['missing']+=stats['missing']*count
                    if stats['valid']:
                        value['_sum']+=stats['mean']*stats['valid']*count
                        value['min']=min(value['min'],stats['min']) if value['min'] is not None else stats['min']
                        value['max']=max(value['max'],stats['max']) if value['max'] is not None else stats['max']
        for group in tables.values():
            for value in group['fields'].values():
                total=value.pop('_sum');value['mean']=total/value['valid'] if value['valid'] else None
                value['sum']=total if value['valid'] and not value['missing'] else None
        derived={}
        for key,group in tables.items():
            if key.startswith('Memory.csv/'):
                fields=group['fields'];size=fields.get('GM_to_UB_datas(KB)',{}).get('sum');n=fields.get('aiv_mte2_instructions',{}).get('sum')
                derived[key+'/bytes_per_mte2']=size*1024/n if size is not None and n else None
        return dict(basic={'Op Name':'multiple_kernels','Device Id':next(iter(devices)),'Task Duration(us)':None},
            metadata=meta,tables=tables,derived=derived,
            kernel_profiles=[dict(occurrences=n,**child) for child,n in children],
            evidence={str(i)+'/'+name:ref for i,(child,_) in enumerate(children) for name,ref in child['evidence'].items()})
    tables, files = {}, {}
    for path in sorted((directory/'profile').rglob('*.csv')):
        name=re.sub(r'_\d{14,}\.csv$', '.csv', path.name)
        if name in tables:
            raise ValueError('Duplicate CSVs on the same side; captures cannot be mixed: '+name)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            tables[name] = list(csv.DictReader(stream))
        files[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    basic = tables.pop('OpBasicInfo.csv', [])
    if len(basic) != 1 or basic[0].get('Op Name') != meta['expected_kernel']:
        raise ValueError('Target kernel mismatch or non-unique basic information')
    if not {'PipeUtilization.csv', 'Memory.csv'}.issubset(tables):
        raise ValueError('Missing pipeline or data-movement records')
    facts = {}
    for table, rows in tables.items():
        keys = [(r.get('block_id'), r.get('sub_block_id')) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate block/sub-block: '+table)
        types = sorted({r.get('sub_block_id', '') for r in rows})
        for core in types:
            group = [r for r in rows if r.get('sub_block_id', '') == core]
            values = {}
            for field in group[0] if group else []:
                if not field or field in ['block_id', 'sub_block_id']:
                    continue
                nums = [number(r.get(field)) for r in group]
                valid = [n for n in nums if n is not None]
                values[field] = dict(valid=len(valid), missing=len(nums)-len(valid),
                    mean=statistics.mean(valid) if valid else None,
                    min=min(valid) if valid else None, max=max(valid) if valid else None,
                    sum=sum(valid) if valid and len(valid) == len(nums) else None)
            facts[table+'/'+core] = dict(rows=len(group), fields=values)
    derived = {}
    for key, group in facts.items():
        if not key.startswith('Memory.csv/'):
            continue
        fields = group['fields']
        nbytes = fields.get('GM_to_UB_datas(KB)', {}).get('sum')
        instr = fields.get('aiv_mte2_instructions', {}).get('sum')
        derived[key+'/bytes_per_mte2'] = nbytes*1024/instr if nbytes is not None and instr and instr > 0 else None
    return dict(basic=basic[0], metadata=meta, tables=facts, derived=derived, evidence=files)


def compare(directory):
    spec_path = directory/'spec.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    raw = spec_path.read_bytes()
    # The first capture was resaved from Windows CRLF to LF; accept only these known byte forms of the same JSON.
    normalized = raw.replace(b'\r\n', b'\n')
    digest = {hashlib.sha256(x).hexdigest() for x in [raw, normalized, normalized.replace(b'\n', b'\r\n')]}
    reference = read_side(directory/'reference', digest, 'reference')
    candidate = read_side(directory/'candidate', digest, 'candidate')
    if reference['metadata']['groups'] != candidate['metadata']['groups']:
        raise ValueError('Metric groups differ')
    for field in ['Device Id']:
        if not reference['basic'].get(field) or reference['basic'].get(field) != candidate['basic'].get(field):
            raise ValueError('Device identity differs or is missing')
    execution_groups = {side: {table: data['rows'] for table, data in facts['tables'].items()}
                        for side, facts in [('reference',reference),('candidate',candidate)]}
    differences = []
    for table in sorted(set(reference['tables']) & set(candidate['tables'])):
        rf, cf = reference['tables'][table]['fields'], candidate['tables'][table]['fields']
        for field in sorted(set(rf) & set(cf)):
            # Compare distribution means for durations/ratios and full row totals for bytes/instructions; retain both distributions.
            aggregation = 'sum' if '(KB)' in field or 'instructions' in field or 'instr_number' in field else 'mean'
            r, c = rf[field][aggregation], cf[field][aggregation]
            if r is None or c is None:
                continue
            differences.append(dict(table=table, metric=field, aggregation=aggregation,
                reference=r, candidate=c, candidate_over_reference=c/r if r else None,
                reference_rows=reference['tables'][table]['rows'],
                candidate_rows=candidate['tables'][table]['rows'],
                scope='Mean of per-block records; differing program counts prevent interpreting this as whole-task critical-path time' if aggregation=='mean'
                      else 'Sum of all recorded counts'))
    result = dict(schema_version='1.0', spec=spec, reference=reference, candidate=candidate,
        differences=differences, derived_differences={}, execution_groups=execution_groups,
        scope='Device OpProf for both implementations on the same input; formal speedup comes from separate KGS results',
        limitations=['Pipelines overlap; their durations do not form an additive latency breakdown', 'Interface bytes do not necessarily equal physical HBM bytes',
            'Separate capture durations cannot be subtracted to obtain exact host overhead', 'Wait/activity ratios support hypotheses requiring source, IR, or experimental verification',
            'Per-record duration means cover different scopes when block/program counts differ; their ratio cannot explain whole-task speedup directly'])
    for key in set(reference['derived']) | set(candidate['derived']):
        r, c = reference['derived'].get(key), candidate['derived'].get(key)
        result['derived_differences'][key] = dict(reference=r, candidate=c,
            candidate_over_reference=c/r if r and c is not None else None)
    return result
