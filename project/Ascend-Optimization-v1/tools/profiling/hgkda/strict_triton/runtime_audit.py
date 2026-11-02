"""只包围候选调用，拒绝PyTorch NPU计算；输入与参考生成在检查范围之外。"""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from common import save,sha

ALLOWED={'aten.empty','aten.empty_like','aten.empty_strided','aten.new_empty','aten.detach','aten.alias','aten.view','aten._unsafe_view',
    'aten.as_strided','aten.slice','aten.select','aten.squeeze','aten.unsqueeze','aten.permute','aten.transpose',
    'aten.t','aten.unbind','aten.split','aten.split_with_sizes','aten.resize_','aten.resize_as_',
    'aten._conj','aten.conj','aten.view_as_real','aten.view_as_complex','aten.is_pinned','aten._pin_memory'}

class CandidateOnly(TorchDispatchMode):
    def __init__(self):super().__init__();self.operations=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=str(func);base='.'.join(name.split('.')[:2])
        npu=any(isinstance(x,torch.Tensor) and x.device.type=='npu' for x in tree_leaves((args,kwargs)))
        # PrivateUse1在此环境通常渲染为npu；同时检查实际注册设备类型。
        npu=npu or any(isinstance(x,torch.Tensor) and x.device.type=='privateuseone' for x in tree_leaves((args,kwargs)))
        record=dict(operation=name,npu_inputs=npu,allowed=base in ALLOWED)
        self.operations.append(record)
        if base not in ALLOWED:
            # 不允许候选在CPU执行Tensor数值计算来替代设备工作；宿主纯标量参数管理不经此接口。
            raise RuntimeError('严格Triton候选触发非许可ATen操作：'+name)
        return func(*args,**kwargs)

def imported(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser()
    for n in ['operator','source','leaf','output']:p.add_argument('--'+n,required=True,type=str if n=='operator' else Path)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(1)
    candidate=imported('strict_runtime_candidate',a.source);oracle=imported('strict_runtime_oracle',a.leaf/'oracle.py')
    cases=[json.loads(x) for x in (a.leaf/'correctness.jsonl').read_text().splitlines() if x.strip()]
    timing=[json.loads(x) for x in (a.leaf/'timing.jsonl').read_text().splitlines() if x.strip()]
    # 原正确性覆盖所有分支，另选最大性能输入；输入本身严格使用原gen_inputs。
    chosen=cases+timing[-1:];rows=[]
    for case in chosen:
        row=dict(case=case['name'],passed=False)
        mode=CandidateOnly()
        try:
            values=oracle.gen_inputs(case,'npu:0')
            with mode:actual=candidate.run(**values)
            torch.npu.synchronize();row['passed']=True
        except Exception as e:row['error']=type(e).__name__+': '+str(e)[:3000]
        row['aten_operations']=mode.operations;rows.append(row)
    source=a.output.parent/('runtime-validator-'+sha(__file__)+'.py');source.write_bytes(Path(__file__).read_bytes())
    result=dict(status='passed' if all(v['passed'] for v in rows) else 'failed',operator=a.operator,
        source_sha256=sha(a.source),checker_source=dict(path=str(source),sha256=sha(source)),
        passed=sum(v['passed'] for v in rows),total=len(rows),cases=rows,
        scope='候选独立运行时检查，不作性能计时或完整数值评测；AST＋依赖审查覆盖未触发分支，trace/IR补充实际设备后端证据')
    save(a.output,result);print(result['status'],result['passed'],result['total'])
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
