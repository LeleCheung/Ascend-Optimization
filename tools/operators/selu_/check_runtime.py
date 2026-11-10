"""SELU专项：核验当前stream、当前指针和未对齐尾块，不改变正式合同。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    import torch,torch_npu
    torch.npu.set_device(0);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('_runtime_candidate',args.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for n in [1024,1009,65536]:
            # 同shape先在默认stream建立缓存；之后切换stream和输入指针。
            candidate.run(torch.full((n,),-.5,dtype=dtype,device='npu:0'))
            torch.npu.synchronize()
            streams=[torch.npu.Stream(),torch.npu.Stream()]
            outputs=[]
            for index,stream in enumerate(streams):
                with torch.npu.stream(stream):
                    base=torch.full((n+2,),7.,dtype=dtype,device='npu:0')
                    x=base[1:-1]
                    x.fill_(-.25 if index==0 else .75)
                    expected=torch.nn.functional.selu(x.clone())
                    result=candidate.run(x)
                    # 跟随读在同一stream上排队；只在全部提交完成后同步。
                    observed=x.clone();finished=torch.npu.Event();finished.record(stream)
                    outputs.append((base,x,result,expected,observed,finished))
            for index,(base,x,result,expected,observed,finished) in enumerate(outputs):
                row=dict(dtype=str(dtype),elements=n,stream_index=index,storage_offset=x.storage_offset())
                try:
                    finished.synchronize()
                    assert result.data_ptr()==x.data_ptr()
                    torch.testing.assert_close(observed,expected,rtol=.015,atol=.015,equal_nan=True)
                    assert base[0].item()==7. and base[-1].item()==7.
                    row['passed']=True
                except Exception as error:
                    row.update(passed=False,error=type(error).__name__+': '+str(error)[:1200])
                rows.append(row)
        # 对齐的快路径在不同stream和新指针上重复使用同一缓存。
        n=65536
        for index in range(2):
            stream=torch.npu.Stream()
            with torch.npu.stream(stream):
                x=torch.full((n,),-.125 if index==0 else 1.25,dtype=dtype,device='npu:0')
                expected=torch.nn.functional.selu(x.clone())
                result=candidate.run(x);observed=x.clone()
                finished=torch.npu.Event();finished.record(stream)
            row=dict(dtype=str(dtype),elements=n,aligned=True,stream_index=index)
            try:
                finished.synchronize();assert result.data_ptr()==x.data_ptr()
                torch.testing.assert_close(observed,expected,rtol=.015,atol=.015,equal_nan=True)
                row['passed']=True
            except Exception as error:
                row.update(passed=False,error=type(error).__name__+': '+str(error)[:1200])
            rows.append(row)
    result=dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,
        scope='当前stream、不同输入指针、未对齐偏移及尾块补测；参考计算只属于检查器')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['passed','total']}))
    if result['passed']!=result['total']:raise SystemExit(1)


if __name__=='__main__':main()
