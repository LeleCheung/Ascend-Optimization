"""以原KGS参数评测独立参考；恢复时同时核验源码、合同和计时配置。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request
from common import PROJECT, load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['operator', 'source', 'leaf', 'output', 'server']:
        parser.add_argument('--'+name, required=True, type=str if name in {'operator', 'server'} else Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--kind', required=True)
    a = parser.parse_args()
    from kernelgen_client.operator_bundles import pack_operator_bundle
    helper = load('cann_priority_evaluate_helper', PROJECT/'tools/evaluation/evaluate-operator.py')
    archive = io.BytesIO()
    digest, _ = pack_operator_bundle(a.leaf, archive)
    bundle_id = 'sha256:' + digest
    name = a.label+'-1'
    a.output.mkdir(parents=True, exist_ok=True)
    request_path = a.output/(name+'.request.json')
    result_path = a.output/(name+'.result.json')
    content = a.source.read_text(encoding='utf8')
    settings = dict(warmup_ms=1000, benchmark_ms=100, num_trials=3, timeout_seconds=1800)
    implementation = dict(name=a.operator+'-'+name, definition=a.operator, language='python',
                          entrypoint='main.py::run', sources=[dict(path='main.py', content=content)])
    if request_path.exists():
        old = json.loads(request_path.read_text(encoding='utf8'))
        assert old['binding']['bundle_id'] == bundle_id, '恢复时参考或合同字节不同'
        assert old['implementation'] == implementation, '恢复时实现身份不同'
        assert old['settings'] == settings, '恢复时计时参数不同'
        if not result_path.exists():
            raise RuntimeError('请求结果未知，禁止重复提交')
        print('EXISTS_VERIFIED', result_path)
        return
    save(a.output/'evaluation-contract.json', dict(source=str(a.source), source_sha256=sha(a.source),
         bundle_id=bundle_id, archive_sha256=digest, kind=a.kind, timing_scope='walltime', settings=settings,
         contract_sha256={n:sha(a.leaf/n) for n in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']}))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(a.server+'/operator-bundles/'+digest, data=archive.getvalue(),
                                 method='PUT', headers={'Content-Type':'application/x-tar'})
    with opener.open(req, timeout=120) as response:
        bundle = json.load(response)
    assert bundle['bundle_id'] == bundle_id and bundle['sha256'] == digest
    save(a.output/'bundle-response.json', bundle)
    payload = dict(api_version='v6.2', binding=dict(bundle_id=bundle_id, definition=a.operator),
                   implementation=implementation, settings=settings)
    save(request_path, payload)
    try:
        value = helper.call(a.server, '/evaluate', payload, 1920)
    except urllib.error.HTTPError as error:
        value = dict(status='HTTP_ERROR', http_status=error.code,
                     detail=error.read().decode('utf8', errors='replace'))
    save(result_path, value)
    print(json.dumps({k:value.get(k) for k in ['status', 'geo_mean', 'num_passed', 'num_workloads', 'is_hack']}, ensure_ascii=False))


if __name__ == '__main__':main()
