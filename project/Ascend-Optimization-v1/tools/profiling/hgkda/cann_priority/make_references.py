"""生成同语义CANN参考，绑定二进制和包装源码身份。"""
from pathlib import Path
from common import BATCH, read, save, sha


def main():
    build = read(BATCH / 'cann-build-v1.json')
    assert sha(build['binary']) == build['binary_sha256']
    prefix = '''import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
'''
    prefix += 'p=Path(' + repr(build['binary']) + ')\nassert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(build['binary_sha256']) + '\n'
    prefix += 's=importlib.util.spec_from_file_location(' + repr(build['module_name']) + ',p)\n_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
    prefix += 'assert torch.__version__==' + repr(build['torch']) + ' and torch._C._GLIBCXX_USE_CXX11_ABI==' + repr(build['abi']) + '\n'
    bodies = {
        'selu_': ('def run(A):\n    return _cpp.inplace(A,0)\n', 'aclnnInplaceSelu'),
        'negative_': ('def run(A):\n    return _cpp.inplace(A,1)\n', 'aclnnInplaceNeg'),
        'log2_': ('def run(A):\n    return _cpp.inplace(A,2)\n', 'aclnnInplaceLog2'),
        'log10_': ('def run(A):\n    return _cpp.inplace(A,3)\n', 'aclnnInplaceLog10'),
        'rad2deg_': ('def run(A):\n    return _cpp.rad2deg(A)\n', 'aclnnInplaceMuls'),
        'nan_to_num_': ('''def run(A,nan=None,posinf=None,neginf=None):
    limits=torch.finfo(A.dtype)
    return _cpp.nan_to_num(A,0. if nan is None else nan,limits.max if posinf is None else posinf,limits.min if neginf is None else neginf)
''', 'aclnnInplaceNanToNum'),
        'cross_out': ('''def run(input,other,dim=None,*,out):
    if dim is None:
        dim=next(i for i,n in enumerate(input.shape) if n==3)
    return _cpp.cross(input,other,dim,out)
''', 'aclnnLinalgCross'),
        'hardshrink_backward': ('def run(grad_out,self_t,lambd=0.5):\n    return _cpp.shrink(grad_out,self_t,lambd)\n', 'aclnnHardshrinkBackward'),
        'subtract_': ('def run(A,B,*,alpha=1):\n    return _cpp.subtract(A,B,alpha)\n', 'aclnnInplaceSub'),
        'round_out': ('''def run(input,*,decimals=0,out=None):
    if out is None:out=torch.empty_like(input)
    return _cpp.round_out(input,decimals,out)
''', 'aclnnRoundDecimals'),
        'special_round_out': ('def run(self,out,*,decimals=0):\n    return _cpp.round_out(self,decimals,out)\n', 'aclnnRoundDecimals'),
        'hardswish_out': ('def run(x,*,out):\n    return _cpp.hardswish_out(x,out)\n', 'aclnnHardswish'),
        'hardswish': ('def run(x):\n    return _cpp.hardswish(x)\n', 'aclnnHardswish'),
        'fmod_tensor_': ('def run(A,B):\n    return _cpp.fmod(A,B)\n', 'aclnnInplaceFmodTensor'),
        'transform_bias_rescale_qkv': ('def run(qkv,qkv_bias,num_heads):\n    return _cpp.qkv(qkv,qkv_bias,num_heads)\n', 'aclnnTransformBiasRescaleQkv'),
        'scatter_add': ('def run(inp,dim,index,src):\n    return _cpp.scatter(inp,dim,index,src)\n', 'aclnnScatterAdd'),
        'upsample_bilinear2d': ('''def run(input,output_size,align_corners=False,scales_h=None,scales_w=None):
    return _cpp.upsample(input,list(output_size),align_corners,0. if scales_h is None else scales_h,0. if scales_w is None else scales_w)
''', 'aclnnUpsampleBilinear2d'),
    }
    index = {}
    for op, (body, api) in bodies.items():
        p = BATCH / 'references' / op / 'main.py'
        content = prefix + body
        if p.exists():
            assert p.read_text(encoding='utf8') == content, '参考已冻结，修改请新建版本'
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf8')
        item = dict(source=str(p), source_sha256=sha(p), api=api, kind='CANN_native_single_api',
                    build=str(BATCH / 'cann-build-v1.json'), build_sha256=sha(BATCH / 'cann-build-v1.json'))
        if op == 'rad2deg_':
            item['scope'] = '专用角度API未找到；单次厂商常量乘法，参数为180/pi'
        index[op] = item
        save(p.parent / 'provenance.json', item)
    previous = BATCH.parent / 'reference-closure-20261031'
    reused = {
        'native_group_norm': (BATCH.parent / 'threeway-20261030/references/native_group_norm/cpp-v3/main.py', 'CANN_native_single_api'),
        'fix': (previous / 'references/fix/main.py', 'CANN_native_single_api'),
        'expm1_': (previous / 'references/expm1_/main.py', 'CANN_dtype_compatibility'),
        'fake_quantize_per_tensor_affine_cachemask_out': (previous / 'references-v2/fake_quantize_per_tensor_affine_cachemask_out/main.py', 'CANN_FP32_compatibility_composition'),
        'batch_norm_with_update_functional': (previous / 'references-v3/batch_norm_with_update_functional/main.py', 'CANN_FP32_state_compatibility_composition'),
    }
    for op, (origin, kind) in reused.items():
        assert origin.is_file()
        p = BATCH / 'references' / op / 'main.py'
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():assert p.read_bytes() == origin.read_bytes()
        else:p.write_bytes(origin.read_bytes())
        item = dict(source=str(p), source_sha256=sha(p), kind=kind, origin=str(origin), origin_sha256=sha(origin),
                    scope='复用已核验参考包装，本批独立复测；不计为新增参考算法')
        index[op] = item
        save(p.parent / 'provenance.json', item)
    save(BATCH / 'reference-index.json', index)
    print('REFERENCES', len(index))


if __name__ == '__main__':
    main()
