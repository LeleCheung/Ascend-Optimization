"""使用KGS原生Python实现类型评测CANN参考，不冒充Triton候选。"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.request
import urllib.error

PROJECT=Path(__file__).resolve().parents[4]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['operator','source','leaf','output','server']:
        p.add_argument('--'+name,required=True,type=str if name in {'operator','server'} else Path)
    p.add_argument('--label',default='CANN-async-python')
    p.add_argument('--kind',default='CANN_native_aclnn')
    a=p.parse_args()
    s=importlib.util.spec_from_file_location('kgs_evaluate_helper',PROJECT/'tools/evaluation/evaluate-operator.py')
    helper=importlib.util.module_from_spec(s);s.loader.exec_module(helper)
    label=a.label+'-1'
    a.output.mkdir(parents=True,exist_ok=True)
    request_path=a.output/(label+'.request.json');result_path=a.output/(label+'.result.json')
    content=a.source.read_text(encoding='utf8')
    if request_path.exists():
        if not result_path.exists():raise RuntimeError('参考请求结果未知，不重发')
        assert json.loads(request_path.read_text())['implementation']['sources'][0]['content']==content
        print('EXISTS '+str(result_path));return
    from kernelgen_client.operator_bundles import pack_operator_bundle
    archive=io.BytesIO();digest,_=pack_operator_bundle(a.leaf,archive)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request=urllib.request.Request(a.server+'/operator-bundles/'+digest,data=archive.getvalue(),method='PUT',headers={'Content-Type':'application/x-tar'})
    with opener.open(request,timeout=120) as response:bundle=json.load(response)
    helper.save(a.output/'bundle-response.json',bundle)
    payload=dict(api_version='v6.2',binding=dict(bundle_id=bundle['bundle_id'],definition=a.operator),
        implementation=dict(name=a.operator+'-'+label,definition=a.operator,language='python',entrypoint='main.py::run',
                            sources=[dict(path='main.py',content=content)]),
        settings=dict(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800))
    helper.save(a.output/'provenance.json',dict(reference_kind=a.kind,implementation_language='python',
        binding_source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),timing_scope='walltime',
        scope='同一原KGS合同和参数；非Triton实现明确使用Python入口类型'))
    helper.save(request_path,payload)
    try:
        result=helper.call(a.server,'/evaluate',payload,1920)
    except urllib.error.HTTPError as error:
        # 保存真实HTTP状态和响应体，区别于连接中断等仍不确定的请求。
        # 明确收到失败响应后也不自动重发这份请求。
        result=dict(status='HTTP_ERROR',http_status=error.code,detail=error.read().decode('utf8',errors='replace'))
    helper.save(result_path,result)
    print(json.dumps({k:result.get(k) for k in ['status','geo_mean','num_passed','num_workloads','is_hack','hack_reason']},ensure_ascii=False))


if __name__=='__main__':main()
