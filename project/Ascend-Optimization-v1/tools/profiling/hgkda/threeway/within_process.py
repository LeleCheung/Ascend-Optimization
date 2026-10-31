"""同进程交替对照log10，逐次恢复输入；补充计时不替换正式KGS。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    import torch_npu
    leaf=a.batch/'catalog/ops/log10_'
    oracle=imported('paired_oracle',leaf/'oracle.py')
    state=json.loads((a.batch/'next-state.json').read_text())
    paths={'HG-KDA-Triton':a.batch/'starting-points/log10_.py'}
    from comparison_sources import native_sources
    paths.update(native_sources(state, 'log10_'))
    reference=state.get('references',{}).get('log10_') or {}
    if reference.get('status')=='PASSED' and reference.get('supplementary_status')=='passed':
        paths['CANN']=Path(reference['source'])
    modules={label:imported('paired_'+str(i),path) for i,(label,path) in enumerate(paths.items())}
    functions={'PyTorch':oracle.run,**{label:module.run for label,module in modules.items()}}
    labels=list(functions)
    result=dict(operator='log10_',status='running',rounds=7,
                schedule='每个case先预热两次，再执行7轮；每轮轮转实现顺序，每次计时前恢复相同输入',
                identities={label:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for label,path in paths.items()},
                clock='time.perf_counter_ns',scope='host_enqueue为目标函数返回前耗时；completion含随后同步等待。KGS正式成绩单列；两者都不是设备单kernel耗时',cases=[])
    cases=[json.loads(x) for x in (leaf/'timing.jsonl').read_text().splitlines() if x.strip()]
    for case in cases:
        values={label:oracle.gen_inputs(case,'npu:0') for label in labels}
        original={label:{name:value.clone() for name,value in inputs.items() if isinstance(value,torch.Tensor)} for label,inputs in values.items()}
        def restore(label):
            for name,pristine in original[label].items():values[label][name].copy_(pristine)
            torch.npu.synchronize()
        expected_inputs=oracle.gen_inputs(case,'npu:0')
        expected=oracle.run(**expected_inputs).clone();torch.npu.synchronize()
        for label in labels:
            for _ in range(2):
                restore(label);actual=functions[label](**values[label]);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)
        samples={label:[] for label in labels}
        for index in range(7):
            for label in labels[index%len(labels):]+labels[:index%len(labels)]:
                restore(label)
                begin=time.perf_counter_ns();actual=functions[label](**values[label]);enqueued=time.perf_counter_ns()
                torch.npu.synchronize();completed=time.perf_counter_ns()
                samples[label].append(dict(host_enqueue_us=(enqueued-begin)/1000.,completion_us=(completed-begin)/1000.))
                torch.testing.assert_close(actual,expected,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)
        summary={label:{metric:dict(median=statistics.median(x[metric] for x in rows),
                    minimum=min(x[metric] for x in rows),maximum=max(x[metric] for x in rows))
                    for metric in ['host_enqueue_us','completion_us']} for label,rows in samples.items()}
        result['cases'].append(dict(case=case,samples=samples,summary=summary,correctness='passed_after_every_call'))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    result['status']='completed'
    result['relative_to_pytorch']={label:{metric:math.exp(statistics.mean(math.log(
        row['summary']['PyTorch'][metric]['median']/row['summary'][label][metric]['median']) for row in result['cases']))
        for metric in ['host_enqueue_us','completion_us']} for label in labels}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for module in modules.values():
        if hasattr(module,'_NATIVE'):module._NATIVE.finish()
    print(json.dumps(dict(status=result['status'],relative_to_pytorch=result['relative_to_pytorch']),ensure_ascii=False))


if __name__=='__main__':main()
