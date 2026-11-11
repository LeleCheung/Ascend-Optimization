"""验证普通调用、无关采集、候选采集及采集后的原地调用。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    import torch
    import torch_npu
    spec=importlib.util.spec_from_file_location('_lifecycle_candidate',a.source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    def check(dtype,stage):
        tensor=torch.linspace(0.25,2.0,1024,device='npu',dtype=dtype)
        expected=torch.log10(tensor)
        pointer=tensor.data_ptr()
        try:
            result=module.run(tensor)
            torch.npu.synchronize()
            assert result is tensor and result.data_ptr()==pointer
            torch.testing.assert_close(result,expected)
            rows.append(dict(case=str(dtype)+'-'+stage,passed=True))
        except Exception as error:
            rows.append(dict(case=str(dtype)+'-'+stage,passed=False,error=type(error).__name__+': '+str(error)))
            raise
    def profile():
        config=torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,l2_cache=False,data_simplification=False)
        return torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,with_stack=False,experimental_config=config)
    try:
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            check(dtype,'ordinary')
            # 已编译候选不在这次采集中执行，验证后来调用也能处理采集状态。
            with profile():
                torch.log10(torch.ones(1024,device='npu',dtype=dtype));torch.npu.synchronize()
            check(dtype,'after-unrelated-profile')
            with profile():check(dtype,'inside-profile')
            check(dtype,'after-profile')
    finally:
        result=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                    total=len(rows),passed=sum(r['passed'] for r in rows),cases=rows,
                    status='passed' if len(rows)==12 and all(r['passed'] for r in rows) else 'failed')
        with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    assert result['status']=='passed'


if __name__=='__main__':main()
