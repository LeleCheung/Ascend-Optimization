"""追加XLogY参考并修复out包装ABI；已提交失败参考保留。"""
import ast
from common import BATCH, read, save, sha


def main():
    index = read(BATCH/'reference-index.json')
    old = BATCH/'references/hardswish_out/main.py'
    body = old.read_text(encoding='utf8').replace('def run(x,*,out):','def run(x,out):')
    out = old.parent/'abi-v2/main.py'
    out.parent.mkdir(exist_ok=True)
    if out.exists():assert out.read_text(encoding='utf8') == body
    else:out.write_text(body, encoding='utf8')
    index['hardswish_out'] = dict(index['hardswish_out'], source=str(out), source_sha256=sha(out),
        scope='仅修复out参数ABI，不改设备库或KGS合同；旧失败结果保留')
    extra = BATCH/'cann-build-extra-v1.json'
    if extra.exists():
        build = read(extra)
        prefix = 'import torch,torch_npu,hashlib,importlib.util\nfrom pathlib import Path\n'
        prefix += 'p=Path('+repr(build['binary'])+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(build['binary_sha256'])+'\n'
        prefix += 's=importlib.util.spec_from_file_location('+repr(build['module_name'])+',p)\n_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
        bodies = {
            'xlogy_scalar_tensor':('def run(self,other):\n    return _cpp.scalar_xlogy(self,other)\n','CANN_native_single_api'),
            'special_xlogy_':('def run(A,B):\n    return _cpp.tensor_xlogy(A,B)\n','CANN_native_single_api'),
            'value_selecting_reduction_backward':('def run(grad,dim,indices,sizes,keepdim):\n    return _cpp.value_backward(grad,dim,indices,list(sizes),keepdim)\n','CANN_zeros_scatter_composition'),
            'cumulative_trapezoid':('def run(y,*,dx=1,dim=-1):\n    return _cpp.trapezoid(y,dx,dim)\n','CANN_Add_Muls_Cumsum_composition'),
        }
        for op,(suffix,kind) in bodies.items():
            target = BATCH/'references'/op/'extra-v1/main.py'
            target.parent.mkdir(parents=True,exist_ok=True)
            text = prefix+suffix
            ast.parse(text)
            if target.exists():assert target.read_text(encoding='utf8') == text
            else:target.write_text(text,encoding='utf8')
            index[op] = dict(source=str(target),source_sha256=sha(target),kind=kind,
                 build=str(extra),build_sha256=sha(extra))
            save(target.parent/'provenance.json',index[op])
    storage = BATCH/'cann-build-storage-rng-v1.json'
    if storage.exists():
        build = read(storage)
        prefix = 'import torch,torch_npu,hashlib,importlib.util\nfrom pathlib import Path\n'
        prefix += 'p=Path('+repr(build['binary'])+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(build['binary_sha256'])+'\n'
        prefix += 's=importlib.util.spec_from_file_location('+repr(build['module_name'])+',p)\n_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
        bodies = {
            'uniform':('def run(self,from_=0.0,to=1.0,*,generator=None):\n    return _cpp.uniform(self,from_,to,generator)\n',
                'CANN_native_uniform_with_generator_adapter','直接厂商随机接口，公开NPU生成器同步状态；旧随机模式counter增量10，实际范围和随机语义须另行核验'),
            '_conj_physical':('def run(input):\n    return _cpp.conj_physical(input)\n',
                'CANN_Copy_Neg_storage_composition','显式DMA或CANN Copy复制实数存储，复数再对虚部调用CANN Neg；处理共轭位，不是专用共轭单kernel'),
        }
        for op,(suffix,kind,note) in bodies.items():
            target = BATCH/'references'/op/'storage-v1/main.py'
            target.parent.mkdir(parents=True,exist_ok=True)
            text = prefix+suffix
            if target.exists():assert target.read_text(encoding='utf8')==text
            else:target.write_text(text,encoding='utf8')
            index[op] = dict(source=str(target),source_sha256=sha(target),kind=kind,
                 build=str(storage),build_sha256=sha(storage),scope=note)
            save(target.parent/'provenance.json',index[op])
    save(BATCH/'reference-index.json',index)
    upsample=BATCH/'references/upsample_bilinear2d/nchw-v2/provenance.json'
    if upsample.exists():
        item=read(upsample)
        assert sha(item['source'])==item['source_sha256']
        index['upsample_bilinear2d']=item
        save(BATCH/'reference-index.json',index)
    print('INDEX',len(index))


if __name__ == '__main__':main()
