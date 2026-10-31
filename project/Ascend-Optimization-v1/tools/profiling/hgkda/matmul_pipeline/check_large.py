"""补充大连续、矩形、边界与非默认stream，确保实际进入新快路径。"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('不覆盖已有补充检查')
    a.output.mkdir(parents=True);assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    s=importlib.util.spec_from_file_location('pipeline_large_candidate',a.source);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    shapes=[(512,1024,512),(768,512,1024),(1024,2048,1024),(529,515,577)]
    rows=[]
    for dt in ['float16','bfloat16','float32']:
        for shape in shapes:
            for strided in [False,True]:
                M,N,K=shape;g=torch.Generator().manual_seed(27019+len(rows))
                def random(s):
                    value=torch.randn(s,generator=g).to(device='npu',dtype=getattr(torch,dt))
                    if strided:
                        store=torch.empty((*s,2),dtype=value.dtype,device=value.device);store[...,1]=value;value=store[...,1]
                    return value
                x,w,b=random((M,K)),random((K,N)),random((N,))
                saved=[v.clone() for v in [x,w,b]];row=dict(dtype=dt,shape=[M,N,K],strided=strided,status='failed');rows.append(row)
                try:
                    reference=torch.relu(x@w+b)
                    stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):actual=m.run(x,w,b)
                    torch.npu.current_stream().wait_stream(stream);torch.npu.synchronize()
                    torch.testing.assert_close(actual,reference,atol=1e-4*K,rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[dt])
                    for value,old in zip([x,w,b],saved):torch.testing.assert_close(value,old,atol=0,rtol=0)
                    if actual.data_ptr() in [x.data_ptr(),w.data_ptr(),b.data_ptr()]:raise ValueError('输出意外别名')
                    row['status']='passed'
                except Exception as e:row['error']=str(e)[-2000:];torch.npu.synchronize()
                result=dict(status='running',source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),cases=rows)
                (a.output/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    result.update(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',passed=sum(r['status']=='passed' for r in rows),total=len(rows))
    (a.output/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    raise SystemExit(result['status']!='passed')

if __name__=='__main__':main()
