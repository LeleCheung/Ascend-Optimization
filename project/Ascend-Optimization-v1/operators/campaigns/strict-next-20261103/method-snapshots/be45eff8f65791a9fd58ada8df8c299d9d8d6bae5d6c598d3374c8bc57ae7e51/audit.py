"""离线核验本批已终态快照；不启动设备，不判断活动搜索已经结束。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(io_path(path).read_bytes()).hexdigest()


def io_path(path):
    return Path('\\\\?\\' + str(path.resolve())) if os.name == 'nt' else path


def main():
    identities = {}
    for path in sorted((BATCH / 'snapshots').glob('fixed-evidence-v*.json')):
        value = read(path)
        for relative, identity in value['files'].items():
            assert relative.startswith('operators/campaigns/strict-next-20261103/')
            target = (PROJECT / relative).resolve()
            assert target.is_relative_to(PROJECT.resolve())
            assert io_path(target).stat().st_size == identity['bytes'] and sha(target) == identity['sha256'], relative
            if relative in identities:
                assert identities[relative] == identity, '快照之间相同路径发生改变'
            identities[relative] = identity
    assert identities, '没有已终态快照'
    old = BATCH.parent / 'strict-five-20261103/delivery.json'
    assert read(BATCH / 'plan.json')['starting_delivery_sha256'] == sha(old)
    assert (BATCH / 'starting-delivery.json').read_bytes() == old.read_bytes()
    assert read(old)['accepted'] == 25
    spec = importlib.util.spec_from_file_location('strict_next_measurements',
                                                 Path(__file__).parent.parent / 'strict_night/measurements.py')
    measurements = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurements)
    folder = BATCH / 'proposals/expm1_/codex-typed-fma-v2'
    result = read(folder / 'result.json')
    assert result['source_sha256'] == sha(folder / 'main.py')
    request = read(folder / 'pytorch/strict-PyTorch-1.request.json')
    sources = request['implementation']['sources']
    assert len(sources) == 1 and sources[0]['content'] == (folder / 'main.py').read_text(encoding='utf8')
    score = measurements.score(read(folder / 'pytorch/strict-PyTorch-1.result.json'))
    assert abs(score - result['PyTorch']) < 1.e-12
    assert score < .6465574073403593 and result['improved'] is False
    assert read(folder / 'extra-pytorch.json')['status'] == 'passed'
    domain = read(folder / 'full-fp16-domain/result.json')
    assert domain['source_sha256'] == result['source_sha256'] and domain['status'] == 'passed'
    assert domain['passed'] == domain['total'] == 4
    assert all(c['matched'] == c['total'] and c['zero_sign_passed'] and c['storage_guards_passed']
               and c['returned_input_object'] for c in domain['cases'])
    compiled = read(BATCH / 'analysis/expm1_/compile-single-buffer-v2/result.json')
    assert compiled['status'] == 'compiled' and len(compiled['rows']) == 6
    assert compiled['source_sha256'] == sha(BATCH / 'compile-probes/single-buffer-layout-v2/main.py')
    assert compiled['device_executed'] is False and compiled['performance_measured'] is False
    value = dict(status='passed', unique_snapshot_files=len(identities),
                 typed_fma_PyTorch=result['PyTorch'], typed_fma_retained=False,
                 buffer_v2='offline_compiled_only', frozen_accepted=25,
                 search_terminal_verified=False, checker_sha256=sha(Path(__file__)),
                 scope='只核验已终态原始附件、正式请求身份与源码、逐case分数、数值门禁及离线编译身份；不以此宣称活动搜索完成或新增达标。')
    (BATCH / 'snapshot-audit.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
