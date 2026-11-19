"""CANNBench官方GQA子集的起点评测；官方误差指标与额外allclose分开。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--candidate',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-only',action='store_true')
    p.add_argument('--precision-floor-only',action='store_true',help='仅在CPU核验输出dtype的最近可表示值能否通过固定公式')
    a=p.parse_args();assert not a.output.exists()
    import torch
    torch.set_num_threads(1)
    if not a.precision_floor_only:
        import torch_npu
        torch.npu.set_device(0)
    contract=json.loads(a.contract.read_bytes())
    official_compare=None
    checker_identity=None
    if contract.get('checker_source'):
        checker_root=Path(contract['checker_source'])
        checker_file=checker_root/'kernel_eval/utils/compare.py'
        checker_identity=dict(path=str(checker_file),sha256=hashlib.sha256(checker_file.read_bytes()).hexdigest(),
                              commit=contract['checker_commit'])
        assert checker_identity['sha256']==contract['checker_sha256']
        sys.path.insert(0,str(checker_root))
        from kernel_eval.utils.compare import compare_tensors
        official_compare=compare_tensors
    golden_sha256=hashlib.sha256(Path(contract['golden']).read_bytes()).hexdigest()
    if contract.get('golden_sha256'):assert contract['golden_sha256']==golden_sha256,'golden reference changed'
    golden_module=load(contract['golden'],'official_gqa')
    golden=golden_module.gqa
    native_reference=getattr(golden_module,'gqa_bench',golden)
    candidate=load(a.candidate,'candidate_gqa').run if a.candidate else None
    rows=[]
    for case in contract['cases']:
        row=dict(case_id=case['case_id'],shape=case['input_shape'],dtype=case['dtype'][0],attrs=case['attrs'])
        try:
            dtype=getattr(torch,case['dtype'][0]);gen=torch.Generator(device='cpu').manual_seed(1729+case['case_id'])
            cpu=[(torch.rand(shape,generator=gen)*2-1).to(dtype) for shape in case['input_shape']]
            expected=golden(*(x.to(torch.float64) for x in cpu),**case['attrs'])
            native_cpu=native_reference(*cpu,**case['attrs']) if official_compare else None
            def correctness(value):
                actual=value.detach().cpu().double();error=(actual-expected).abs();relative=error/(expected.abs()+1e-7)
                threshold=2**(-10 if dtype==torch.float16 else -7)
                mere=relative.mean().item();mare=relative.max().item()
                result=dict(mere=mere,mare=mare,threshold=threshold,max_abs=error.max().item(),
                    official_formula_pass=mere<threshold and mare<10*threshold,
                    supplementary_allclose=bool(torch.allclose(actual,expected,atol=.015,rtol=.015)))
                if official_compare:
                    checked=official_compare(output=value.detach().cpu(),golden=expected,
                        dtype=case['dtype'][0],threshold=threshold,native_output=native_cpu)
                    result['official_checker']=checked.to_dict()
                return result
            row['nearest_representable_correctness']=correctness(expected.to(dtype))
            if a.precision_floor_only:
                row['status']='precision_floor_only';rows.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
                continue
            inputs=[x.to('npu:0') for x in cpu];kwargs=case['attrs'];scale=kwargs['scaleValue']
            if scale<=0:scale=inputs[0].shape[-1]**-.5
            sq,sk=inputs[0].shape[1],inputs[1].shape[1]
            mask=None
            if kwargs['is_causal']:
                mask=(torch.arange(sk)[None,:]>torch.arange(sq)[:,None]+sk-sq).to('npu:0')
            def native():
                return torch.ops.npu.npu_fused_infer_attention_score(*inputs,num_heads=inputs[0].shape[2],
                    num_key_value_heads=inputs[1].shape[2],input_layout='BSND',scale=scale,atten_mask=mask,sparse_mode=0)[0]
            def measure(fn):
                for _ in range(contract['warmup']):fn()
                torch.npu.synchronize();times=[]
                for _ in range(contract['trials']):
                    start=time.perf_counter_ns()
                    for _ in range(contract['iterations']):fn()
                    torch.npu.synchronize();times.append((time.perf_counter_ns()-start)/1000/contract['iterations'])
                return dict(full_call_us=sorted(times)[len(times)//2],trials_us=times,
                            method='CPU wall clock averaged over a batch, synchronization at batch boundary')
            def invoke():return candidate(*inputs,**kwargs)
            functions=[('native',native)]
            if candidate and not a.reference_only:functions.append(('candidate',invoke))
            for label,fn in functions:
                try:
                    value=fn();torch.npu.synchronize();row[label+'_correctness']=correctness(value)
                    row[label+'_timing']=measure(fn)
                except Exception as error:
                    row[label+'_error']=type(error).__name__+': '+str(error)
            if 'native_timing' in row and 'candidate_timing' in row:
                row['native_over_candidate']=row['native_timing']['full_call_us']/row['candidate_timing']['full_call_us']
            row['status']='failed' if any(k.endswith('_error') for k in row) else 'measured'
        except Exception as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error))
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    result=dict(schema='hgkda.cannbench-pilot/v1',task='level4/gqa',scope='explicit official case subset; not full CANNBench acceptance or leaderboard',
        contract_sha256=hashlib.sha256(a.contract.read_bytes()).hexdigest(),
        golden_sha256=golden_sha256,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_checker_identity=checker_identity,
        source_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest() if a.candidate else None,cases=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(cases=len(rows),measured=sum(r['status']=='measured' for r in rows),output=str(a.output))))


if __name__=='__main__':main()
