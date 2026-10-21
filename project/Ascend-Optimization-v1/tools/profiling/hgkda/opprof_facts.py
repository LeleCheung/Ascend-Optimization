"""读取已归档的CANN OpProf设备计数，保留block口径，不推算硬件峰值。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile


def _table(raw):
    rows = [row for row in csv.reader(io.StringIO(raw.decode('utf-8-sig'))) if any(row)]
    if not rows:
        raise ValueError('OpProf CSV为空')
    header = rows[0]
    names = [name for name in header if name]
    if len(set(names)) != len(names) or any(len(row) != len(header) for row in rows[1:]):
        raise ValueError('OpProf CSV列名重复或列数不一致')
    return [{key: value for key, value in zip(header, row) if key} for row in rows[1:]]


def _number(value):
    if value in ('', 'NA', 'N/A', None):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _summarize(rows):
    keys = [(row['block_id'], row['sub_block_id']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('OpProf block/sub-block重复，不能混合不同采集')
    numeric = {}
    for name in rows[0] if rows else []:
        if name in ('block_id', 'sub_block_id'):
            continue
        values = [value for row in rows if (value := _number(row.get(name))) is not None]
        numeric[name] = {'valid_rows': len(values), 'missing_or_non_numeric_rows': len(rows)-len(values),
                         'min': min(values) if values else None,
                         'max': max(values) if values else None,
                         'mean': statistics.mean(values) if values else None,
                         'median': statistics.median(values) if values else None}
    return {'rows': len(rows), 'sub_blocks': sorted({key[1] for key in keys}),
            'aggregation': 'unweighted per-block/sub-block distribution; not device utilization',
            'numeric': numeric}


def decode(archive_path, manifest, *, expected_archive_sha256, expected_solution_sha256,
           expected_case_id, expected_kernel, kernel_index=0):
    archive_path = Path(archive_path)
    if archive_path.stat().st_size > 64*1024*1024:
        raise ValueError('OpProf压缩包超过64MiB')
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_sha != expected_archive_sha256:
        raise ValueError('OpProf压缩包SHA不匹配')
    request, result = manifest['request'], manifest['result']
    if (request['identity']['solution_sha256'] != expected_solution_sha256
            or request['case_id'] != expected_case_id):
        raise ValueError('OpProf候选或用例身份不匹配')
    artifacts = [a for a in result.get('artifacts', []) if a.get('kind') == 'vendor_kernel_report'
                 and a.get('filename') == 'msprof-op-report.tar.gz']
    if len(artifacts) != 1 or artifacts[0].get('size_bytes') != archive_path.stat().st_size:
        raise ValueError('OpProf附件类型、数量或长度不匹配')
    tables, hashes = {}, {}
    total = count = 0
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for member in archive:
            count += 1;total += max(0, member.size)
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or member.issym() or member.islnk()
                    or count > 2048 or total > 64*1024*1024 or member.size < 0):
                raise ValueError('OpProf归档路径、类型或大小不合法')
            if not member.isfile() or not path.parts or path.parts[0] != 'kernel-%03d' % kernel_index:
                continue
            if path.suffix != '.csv':
                continue
            if member.size > 8*1024*1024 or path.name in tables:
                raise ValueError('OpProf表过大或同kernel内表名重复')
            raw = archive.extractfile(member).read()
            tables[path.name] = _table(raw)
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
    basic = tables.pop('OpBasicInfo.csv', [])
    if len(basic) != 1 or basic[0].get('Op Name') != expected_kernel:
        raise ValueError('OpProf kernel身份不同或基本信息不唯一')
    if not {'Memory.csv', 'PipeUtilization.csv', 'L2Cache.csv'}.issubset(tables):
        raise ValueError('OpProf缺少设备计数表')
    output = {'schema_version': '1.0', 'profile_id': result['profile_id'],
              'case_id': expected_case_id, 'solution_sha256': expected_solution_sha256,
              'profile_status': result.get('status'), 'collection_error': result.get('error'),
              'partial_collection': result.get('status') != 'completed',
              'instruction_listing_available': any(a.get('kind') == 'instruction_listing'
                                                    for a in result.get('artifacts', [])),
              'archive_sha256': archive_sha, 'kernel_index': kernel_index,
              'kernel': basic[0], 'member_sha256': hashes,
              'tables': {name: _summarize(rows) for name, rows in tables.items()},
              'memory_reported_totals_KB': {}, 'l2_read_hit_rate_pct': {},
              'scope': 'OpProf device-counter collection, separate from Eval and base msprof metrics',
              'limitations': ['Block-level busy/usage ratios cannot be added into a device time breakdown.',
                              'Reported memory-interface counters are not automatically physical HBM traffic.',
                              'Vendor bandwidth usage fields retain their original units and normalization.',
                              'Device OpProf completion does not imply simulator instruction collection succeeded.']}
    memory = tables['Memory.csv']
    for field in ['read_main_memory_datas(KB)', 'write_main_memory_datas(KB)',
                  'GM_to_UB_datas(KB)', 'UB_to_GM_datas(KB)']:
        values = [_number(row.get(field)) for row in memory]
        output['memory_reported_totals_KB'][field] = (
            sum(values) if values and all(value is not None for value in values) else None)
    for prefix in ['aic', 'aiv']:
        fields = [prefix+'_'+channel+'_read_cache_'+outcome
                  for channel in ['r0', 'r1'] for outcome in ['hit', 'miss_allocate']]
        values = [[_number(row.get(field)) for field in fields] for row in tables['L2Cache.csv']]
        if values and all(all(value is not None and value >= 0 for value in row) for row in values):
            hits = sum(row[0]+row[2] for row in values)
            misses = sum(row[1]+row[3] for row in values)
            output['l2_read_hit_rate_pct'][prefix] = {
                'hits': hits, 'misses': misses,
                'weighted_hit_rate_pct': 100*hits/(hits+misses) if hits+misses else None,
                'method': 'sum read hits / sum(read hits + read miss allocate); not mean of percentages'}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    parser.add_argument('--expected-solution-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--kernel-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出已存在，使用新文件保留原始证据')
    result = decode(args.archive, json.loads(args.manifest.read_text(encoding='utf-8-sig')),
                    expected_archive_sha256=args.expected_archive_sha256,
                    expected_solution_sha256=args.expected_solution_sha256,
                    expected_case_id=args.case_id, expected_kernel=args.kernel,
                    kernel_index=args.kernel_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'tables': len(result['tables']),
                      'profile_status': result['profile_status']}))


if __name__ == '__main__':main()
