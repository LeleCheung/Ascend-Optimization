"""核对固定包与单文件入口，使用同一合同输入区分布局和适配问题。"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

from driver import imported


def check(batch, output):
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    if output.exists():
        raise FileExistsError(output)
    import torch
    import torch_npu
    import flag_gems
    rows=[]
    for op in ['negative_', 'expm1_']:
        leaf=batch/'catalog/ops'/op
        oracle=imported('probe_oracle_'+op,leaf/'oracle.py')
        copied=imported('probe_copied_'+op,batch/'seeds'/(op+'.py'))
        package=importlib.import_module('flag_gems.ops.'+('negative_' if op=='negative_' else 'expm1'))
        functions={'oracle':oracle.run,'copied':copied.run,'fixed_package':getattr(package,op),'public_package':getattr(flag_gems,op)}
        cases=[json.loads(line) for line in (leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
        selected=[c for c in cases if c['inputs']['spec']['dtype']=='float32' and c['inputs']['spec']['shape']==[7,9]]
        for case in selected:
            for side,fn in functions.items():
                args=oracle.gen_inputs(case,'npu:0');x=args['A'];torch.npu.synchronize()
                before=x.cpu();expected=-before if op=='negative_' else torch.expm1(before)
                path=Path(inspect.getsourcefile(fn))
                row=dict(operator=op,case=case['name'],side=side,input_stride=list(x.stride()),source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),function_module=fn.__module__,status='failed')
                try:
                    actual=fn(**args);torch.npu.synchronize()
                    row.update(output_stride=list(actual.stride()),alias=torch._C._is_alias_of(actual,x),actual_sample=actual.cpu().flatten()[:10].tolist())
                    tol=case['tolerance'];torch.testing.assert_close(actual.cpu(),expected,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    assert row['alias']
                    row['status']='passed'
                except Exception as error:
                    row['error']=type(error).__name__+': '+str(error)[:1800]
                rows.append(row)
    data=dict(status='completed',cases=rows,scope='同一Native合同的两种FP32布局，与CPU数学参考核对；保留复制入口、固定模块与公共入口的实际函数身份')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps([dict(operator=r['operator'],case=r['case'],side=r['side'],status=r['status']) for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();check(a.batch,a.output)
