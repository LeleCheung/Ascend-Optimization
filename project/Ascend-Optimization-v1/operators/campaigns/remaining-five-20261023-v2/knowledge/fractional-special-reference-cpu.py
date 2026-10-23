"""只用CPU核对分数池化特殊值与单点输出索引规则。"""
from pathlib import Path
import hashlib,json
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
    samples=torch.tensor([[[.13,.37,.79]]],dtype=dtype)
    for size in [(2,2,2),(1,1,1)]:
        shape=(1,1,5,5,5)
        inputs={
            'ascending':torch.arange(125,dtype=torch.float32).reshape(shape).to(dtype),
            'negative_inf':torch.full(shape,-float('inf'),dtype=dtype),
            'all_nan':torch.full(shape,float('nan'),dtype=dtype),
            'ties':torch.ones(shape,dtype=dtype),
        }
        for pattern,x in inputs.items():
            values,indices=F.fractional_max_pool3d(x,(2,2,2),output_size=size,
                                                 return_indices=True,_random_samples=samples)
            rows.append(dict(dtype=str(dtype),output_size=size,pattern=pattern,
                             indices=indices.flatten().tolist(),
                             nan_count=int(values.isnan().sum()),
                             negative_inf_count=int(values.isneginf().sum())))
result=dict(scope='本机PyTorch的CPU语义探针，不是候选NPU数值或性能验证',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows)
path=Path(__file__).with_name('fractional-special-reference-cpu.json')
assert not path.exists(),path
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
