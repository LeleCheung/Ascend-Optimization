#!/usr/bin/env python3
"""比较同spec的两份OpProf采集；整理事实，不以活跃比例自动推断因果。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_side(directory, expected_spec_sha, side):
    meta = json.loads((directory/'collection.json').read_text())
    if meta['status'] != 'completed' or meta['spec_sha256'] not in expected_spec_sha or meta['side'] != side:
        raise ValueError('采集身份或完成状态不符：'+str(directory))
    tables, files = {}, {}
    for path in sorted((directory/'profile').rglob('*.csv')):
        if path.name in tables:
            raise ValueError('同侧存在重复CSV，不能混合采集：'+path.name)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            tables[path.name] = list(csv.DictReader(stream))
        files[path.name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    basic = tables.pop('OpBasicInfo.csv', [])
    if len(basic) != 1 or basic[0].get('Op Name') != meta['expected_kernel']:
        raise ValueError('目标kernel不同或基本信息不唯一')
    if not {'PipeUtilization.csv', 'Memory.csv'}.issubset(tables):
        raise ValueError('缺少流水或搬运数据')
    facts = {}
    for table, rows in tables.items():
        keys = [(r.get('block_id'), r.get('sub_block_id')) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('重复block/sub-block：'+table)
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
    spec = json.loads(spec_path.read_text())
    raw = spec_path.read_bytes()
    # 第一份采集将Windows的CRLF重存为LF；只允许这两种已知字节表示，JSON内容不变。
    normalized = raw.replace(b'\r\n', b'\n')
    digest = {hashlib.sha256(x).hexdigest() for x in [raw, normalized, normalized.replace(b'\n', b'\r\n')]}
    reference = read_side(directory/'reference', digest, 'reference')
    candidate = read_side(directory/'candidate', digest, 'candidate')
    if reference['metadata']['groups'] != candidate['metadata']['groups']:
        raise ValueError('指标组不同')
    for field in ['Device Id']:
        if not reference['basic'].get(field) or reference['basic'].get(field) != candidate['basic'].get(field):
            raise ValueError('设备身份不同或缺失')
    execution_groups = {side: {table: data['rows'] for table, data in facts['tables'].items()}
                        for side, facts in [('reference',reference),('candidate',candidate)]}
    differences = []
    for table in sorted(set(reference['tables']) & set(candidate['tables'])):
        rf, cf = reference['tables'][table]['fields'], candidate['tables'][table]['fields']
        for field in sorted(set(rf) & set(cf)):
            # 时长/比例只比较分布均值；字节/指令使用完整行总数，同时保留各侧原始分布。
            aggregation = 'sum' if '(KB)' in field or 'instructions' in field or 'instr_number' in field else 'mean'
            r, c = rf[field][aggregation], cf[field][aggregation]
            if r is None or c is None:
                continue
            differences.append(dict(table=table, metric=field, aggregation=aggregation,
                reference=r, candidate=c, candidate_over_reference=c/r if r else None,
                reference_rows=reference['tables'][table]['rows'],
                candidate_rows=candidate['tables'][table]['rows'],
                scope='逐block分布均值；两侧program数不同时，不解释为全任务关键路径占时' if aggregation=='mean'
                      else '完整记录计数总和'))
    result = dict(schema_version='1.0', spec=spec, reference=reference, candidate=candidate,
        differences=differences, derived_differences={}, execution_groups=execution_groups,
        scope='同输入两侧设备OpProf；正式加速比使用独立KGS结果',
        limitations=['各流水可重叠，不相加为耗时分解', '接口字节不自动等于物理HBM字节',
            '独立采集时间不相减为精确host耗时', '等待与活跃比例仅形成假设；需源码/IR或实验核验',
            '不同block/program数下每记录时间的均值范围不同，不能直接用其比值归因整任务加速'])
    for key in set(reference['derived']) | set(candidate['derived']):
        r, c = reference['derived'].get(key), candidate['derived'].get(key)
        result['derived_differences'][key] = dict(reference=r, candidate=c,
            candidate_over_reference=c/r if r and c is not None else None)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('输出已存在，请保留旧结果')
    a.output.write_text(json.dumps(compare(a.directory), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
