"""逐字节核验30项原合同、搜索合同与起点，区分合同核验和性能达标。"""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
BATCH = PROJECT/'operators/campaigns/strict-night-20261102'
OLD = BATCH.parent/'strict-triton-20261102'
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'
FILES = ('definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(name):
    assert str(name).startswith(REMOTE), '只接受项目内的冻结来源'
    path = PROJECT/str(name)[len(REMOTE):]
    path.resolve().relative_to(PROJECT.resolve())
    return path


def main():
    plan = read(BATCH/'plan.json')
    previous = read(BATCH/'starting-delivery.json')
    delivery = read(BATCH/'delivery.json')
    references = read(OLD/'reference-qualification-audit.json')
    assert sha(OLD/'delivery.json') == sha(BATCH/'starting-delivery.json') == plan['starting_delivery_sha256']
    assert previous['accepted'] == 16 and plan['candidate_backend'] == 'strict_Triton'
    assert plan['physical_card'] == 7 and plan['timing_scope'] == 'walltime'
    assert references['status'] == 'verified' and references['software_lock_verified']
    assert references['qualified'] == 25 and references['total'] == 30
    frozen = {row['operator']: row for row in previous['rows']}
    qualified = {row['operator']: row['qualified'] for row in references['rows']}
    names = [row['operator'] for row in plan['records']]
    assert names == [row['operator'] for row in previous['rows']] == [row['operator'] for row in delivery['rows']]
    assert len(names) == len(set(names)) == 30
    evidence = {}
    rows = []
    for record in plan['records']:
        op = record['operator']
        old = frozen[op]
        is_cann = bool(old.get('cann'))
        assert is_cann == bool(qualified[op])
        assert record['reference'] == ('qualified_CANN' if is_cann else 'original_PyTorch_only')
        original = OLD/'catalog/ops'/op
        search = OLD/('cann-catalog' if is_cann else 'catalog')/'ops'/op
        for kind, origin in [('original-catalog', original), ('catalog', search)]:
            for name in FILES:
                source = origin/name
                current = BATCH/kind/'ops'/op/name
                assert current.read_bytes() == source.read_bytes(), f'冻结合同变化：{op}/{kind}/{name}'
                if kind == 'catalog':
                    assert sha(current) == record['contract_sha256'][name]
                for file in [source, current]:
                    evidence[file.relative_to(PROJECT).as_posix()] = sha(file)
        start = BATCH/record['seed']
        assert start.read_bytes() == local(old['source']).read_bytes(), '起点源码变化：'+op
        assert sha(start) == record['source_sha256'] == old['source_sha256']
        evidence[start.relative_to(PROJECT).as_posix()] = sha(start)
        rows.append(dict(operator=op, group=old['group'], reference=record['reference'],
                         original_contract_files=4, search_contract_files=4,
                         starting_source_sha256=sha(start)))
    result = dict(status='verified', operators=30, contract_file_comparisons=240,
                  starting_sources_verified=30, reference_identity_verified=30,
                  input_tolerance_reference_and_timing_files_unchanged=True,
                  frozen_accepted=16, current_accepted=delivery['accepted'],
                  rows=rows, evidence_sha256=evidence,
                  scope='只核验已保存合同与起点身份；不执行设备评测，不把合同核验通过解释为30项性能均达标。')
    output = BATCH/'analysis/frozen-contract-audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps({key: value for key, value in result.items() if key not in {'rows','evidence_sha256','scope'}}))


if __name__ == '__main__':
    main()
