"""同数学内核的分块、调度与累积精度变换；未知布局保留父路径。"""
import argparse
import hashlib
import json
from pathlib import Path
import ast

MATMUL=r'''

_device_parent = run
def run(input, weight, bias):
    m,k=input.shape;n=weight.shape[1]
    bm,bn,bk=(__BM__,__BN__,__BK32__) if input.dtype==torch.float32 else (__BM__,__BN__,__BK16__)
    if not (m>0 and n>0 and k>0 and m%bm==0 and n%(2*bn)==0 and k%bk==0
            and input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
            and bias.numel()==n):
        return _device_parent(input,weight,bias)
    out=torch.empty((m,n),dtype=input.dtype,device=input.device)
    kernel=hg_device_fp32_kernel if input.dtype==torch.float32 else hg_device_half_kernel
    with torch_device_fn.device(input.device):
        kernel[((m//bm)*(n//(2*bn)),)](input,weight,bias.reshape(-1),out,m,n,k,
            input.stride(0),input.stride(1),weight.stride(0),weight.stride(1),
            bias.reshape(-1).stride(0),out.stride(0),out.stride(1),bm,bn,bk,__GROUP__,
            multibuffer=__MULTI__,unit_flag=True,sync_solver=False)
    return out
matmul_bias_activation=run
'''

AMIN=r'''

_device_parent=run
def run(inp,dim=None,keepdim=False):
    dims=[dim] if isinstance(dim,int) else dim
    if not (inp.is_contiguous() and inp.ndim in (2,3) and dims is not None
            and len(dims)==1 and dims[0]%inp.ndim==1 and inp.shape[1]>0
            and inp.numel()>0):
        return _device_parent(inp,dim=dim,keepdim=keepdim)
    shape=list(inp.shape);out_shape=shape[:]
    if keepdim:out_shape[1]=1
    else:out_shape.pop(1)
    out=torch.empty(out_shape,dtype=inp.dtype,device=inp.device)
    native=__NATIVE__ and inp.dtype==torch.float16
    with torch_device_fn.device(inp.device):
        if inp.ndim==2:
            bo,br=__BO__,__LAST_BR__
            masked=shape[0]%bo!=0 or shape[1]%br!=0
            amin_native_last_kernel[(triton.cdiv(shape[0],bo),)](inp,out,shape[0],shape[1],bo,br,native,masked,multibuffer=__MULTI__)
        else:
            bc,br=__BC__,__MIDDLE_BR__
            masked=shape[2]%bc!=0 or shape[1]%br!=0
            amin_native_middle_kernel[(shape[0],triton.cdiv(shape[2],bc))](inp,out,shape[1],shape[2],bc,br,native,masked,multibuffer=__MULTI__)
    return out
amin=run
'''

DEFAULTS={
 'matmul_bias_activation':dict(bm=128,bn=128,bk32=128,bk16=256,group=0,multi=True),
 'amin':dict(bo=8,last_br=1024,bc=512,middle_br=16,native=True,multi=True)}


def validate(op,cfg):
    if set(cfg)!=set(DEFAULTS[op]):raise ValueError('配置键不完整')
    for key,value in cfg.items():
        if isinstance(DEFAULTS[op][key],bool):
            if type(value) is not bool:raise ValueError('布尔配置类型不同')
        elif type(value) is not int or value not in {
            'bm':[32,64,128],'bn':[32,64,128],'bk32':[32,64,128,256],
            'bk16':[64,128,256,512],'group':[0,2,4,8], 'bo':[2,4,8,16,32],
            'last_br':[128,256,512,1024,2048], 'bc':[64,128,256,512,1024],
            'middle_br':[4,8,16,32,64]}[key]:raise ValueError('超出冻结配置空间：'+key)


def clone(source,name,newname):
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines();start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    text='\n'.join(lines[start-1:node.end_lineno])
    text=text.replace('def '+name+'(','def '+newname+'(',1)
    text=text.replace('    BLOCK_SIZE_K: tl.constexpr,\n','    BLOCK_SIZE_K: tl.constexpr,\n    HG_GROUP: tl.constexpr,\n',1)
    old='    pid_m = pid % num_pid_m\n    pid_n = pid // num_pid_m'
    new='''    if HG_GROUP == 0:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_n = tl.cdiv(N, 2 * BLOCK_SIZE_N)
        width = HG_GROUP * num_pid_n
        first = (pid // width) * HG_GROUP
        size = tl.minimum(num_pid_m - first, HG_GROUP)
        pid_m = first + pid % size
        pid_n = (pid % width) // size'''
    if old not in text:raise ValueError('父内核调度结构不匹配')
    return text.replace(old,new,1)


def transform(source,op,cfg):
    validate(op,cfg)
    if op=='matmul_bias_activation':
        text=source+'\n\n'+clone(source,'mba_fp32_dual_v9_kernel','hg_device_fp32_kernel')+'\n\n'
        text+=clone(source,'mba_dual_dispatch_v8_kernel','hg_device_half_kernel')+'\n'+MATMUL
    else:text=source+'\n'+AMIN
    for k,v in cfg.items():text=text.replace('__'+k.upper()+'__',repr(v))
    ast.parse(text);return text


def probes(op):
    base=DEFAULTS[op]
    choices=({'compact_m':dict(bm=64),'compact_n':dict(bn=64),'larger_k':dict(bk32=256,bk16=512),
              'group4':dict(group=4),'no_multibuffer':dict(multi=False)} if op=='matmul_bias_activation' else
             {'wider_transfer':dict(bc=1024,middle_br=8),'more_rows':dict(bo=16,last_br=512),
              'no_multibuffer':dict(multi=False),'fp32_accumulator':dict(native=False),
              'short_reduction_tile':dict(middle_br=8,last_br=512)})
    return [('reexpress_parent',base)]+[(key,{**base,**change}) for key,change in choices.items()]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('候选已存在')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes(transform(a.source.read_text(encoding='utf8'),a.operator,json.loads(a.config.read_text())).encode())
