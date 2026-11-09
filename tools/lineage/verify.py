"""离线复核三类版本原件、源码身份和汇报范围；不运行设备或模型。"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(index, repository):
    seen = set()

    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                p = Path(value['path'])
                if not p.is_absolute():
                    path = repository/p
                    assert path.is_file(), p
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], p
                    seen.add(p.as_posix())
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    rows = index['rows']
    assert len(rows) == index['count'] == 39
    assert len({r['operator'] for r in rows}) == 39
    assert {group: sum(r['group'] == group for r in rows) for group in ['1015', '1029', 'early']} == dict(
        **{'1015': 10, '1029': 20, 'early': 9})
    assert sum(r['current_original_contract_accepted'] is True for r in rows) == 26
    for row in rows:
        versions = row['versions']
        assert {'historical_nonconforming', 'historical_strict_triton', 'independent_entry_verified'} <= versions.keys()
        if row['group'] != 'early':
            strict = versions['historical_strict_triton']
            source_sha = strict['source']['sha256']
            review = json.loads((repository/strict['review']['path']).read_bytes())
            assert review['source_sha256'] == source_sha
            for evidence in strict['evidence'].values():
                if evidence.get('source_sha256'):
                    assert evidence['source_sha256'] == source_sha
            mixed = versions['historical_nonconforming']
            for measured in mixed.get('measurements', {}).values():
                if measured and measured.get('verified_source_binding'):
                    assert measured['binding']['frozen_source_sha256'] == mixed['source']['sha256']
            latest=versions['independent_entry_verified']
            generated=([latest['generated_experiment']] if latest.get('generated_experiment') else [])
            for experiment in generated+latest.get('generated_history',[]):
                result=json.loads((repository/experiment['formal_result']['path']).read_bytes())
                review=json.loads((repository/experiment['review']['path']).read_bytes())
                audit=json.loads((repository/experiment['audit']['path']).read_bytes())
                matches=[r for r in audit['rows'] if r['source_sha256']==experiment['source']['sha256']]
                assert audit['evidence_verified'] and len(matches)==1
                assert review['source_sha256']==experiment['source']['sha256']
                assert result['geo_mean']==matches[0]['geo_mean']==experiment['cann_score']
                assert matches[0]['accepted']==experiment['accepted']
        walk(row)
    walk(index['artifacts'])
    return dict(status='passed', operators=39, original_contract_accepted=26,
                unique_files_checked=len(seen),
                nonconforming_sources=sum(v['versions']['historical_nonconforming']['status'] ==
                                         'archived_nonconforming' for v in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(json.loads(args.index.read_bytes()), args.repository.resolve())
    if args.output:
        assert not args.output.exists()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
