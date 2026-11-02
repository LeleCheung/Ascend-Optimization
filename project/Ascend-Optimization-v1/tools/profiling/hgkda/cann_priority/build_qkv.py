"""单CPU编译QKV宿主桥；计算保持原Triton kernel，设备验证另由队列执行。"""
import argparse
import os
from pathlib import Path
import shutil
from common import BATCH,HERE,ROOT,PY,read,save,sha,load


def build():
    record=BATCH/'qkv-host-build-v1.json'
    if record.exists():
        for p,digest in read(record)['files'].items():assert sha(p)==digest
        return
    os.environ['MAX_JOBS']='1'
    import torch,torch_npu
    from torch.utils.cpp_extension import load as compile_host
    folder=BATCH/'native-qkv-host-v1';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'qkv_host.cpp';source.write_bytes((HERE/'qkv_host.cpp').read_bytes())
    npu=Path(torch_npu.__file__).parent
    build_dir=ROOT/'runtime/cann-priority-qkv-host-v1';build_dir.mkdir(parents=True,exist_ok=True)
    m=compile_host(name='_hgkda_qkv_host_v1',sources=[str(source)],build_directory=str(build_dir),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
    binary=folder/Path(m.__file__).name;shutil.copyfile(m.__file__,binary)
    save(record,dict(status='built',binary=str(binary),files={str(p):sha(p) for p in [source,binary]},
         torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI))


def proposal():
    info=read(BATCH/'qkv-host-build-v1.json')
    for p,digest in info['files'].items():assert sha(p)==digest
    out=BATCH/'proposals/transform_bias_rescale_qkv/cpp-host-v1';out.mkdir(parents=True,exist_ok=True)
    files={}
    for p in info['files']:
        p=Path(p);target=out/p.name
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copyfile(p,target)
        files[str(target)]=sha(target)
    parent=BATCH/'starting-points/transform_bias_rescale_qkv.py'
    binary=out/Path(info['binary']).name
    code=parent.read_text(encoding='utf8')+'\n\n_original_run=run\n'
    code+='import hashlib as _hashlib,importlib.util as _importlib\nfrom pathlib import Path as _Path\n'
    code+='_host_path=_Path('+repr(str(binary))+')\nassert _hashlib.sha256(_host_path.read_bytes()).hexdigest()=='+repr(sha(binary))+'\n'
    code+='assert torch.__version__=='+repr(info['torch'])+' and torch._C._GLIBCXX_USE_CXX11_ABI=='+repr(info['abi'])+'\n'
    code+='''_spec=_importlib.spec_from_file_location("_hgkda_qkv_host_v1",_host_path)
_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)

def _compile_for_host(x,bias,q,k,v,constants):
    compiled=_qkv_tile.warmup(x,bias,q,k,v,*constants,grid=(1,1))
    launcher=compiled.run
    if (getattr(launcher,'compile_only',False) or getattr(launcher,'enable_msprof_register_tensor',False) or
        getattr(launcher.metadata,'debug_enabled',False) or getattr(launcher.metadata,'debug',False)):
        return None
    return launcher.launch,compiled.function,compiled.packed_metadata

def run(qkv,qkv_bias,num_heads):
    return _host.execute(qkv,qkv_bias,num_heads,_original_run,_compile_for_host)
'''
    source=out/'main.py'
    if source.exists():assert source.read_text(encoding='utf8')==code
    else:source.write_text(code,encoding='utf8')
    files[str(source)]=sha(source)
    attribution='Codex将既有KG Triton QKV实现的分配、元数据和参数打包移到C++；Triton数学计算不变，非对齐及采集模式保留原路径'
    save(out/'manifest.json',dict(files=files,source_sha256=sha(source),host_build=info,attribution=attribution))
    save(out/'proposal.json',dict(operator='transform_bias_rescale_qkv',id=out.name,source='main.py',source_sha256=sha(source),
        backend='Triton_with_Cpp_host',attribution=attribution,
        prediction='本批相对CANN为0.785；保持设备计算，减少Python shape/stride/data_ptr/分配和参数打包成本；完整合同及57项补测验证'))
    print('QKV_PROPOSAL',sha(source))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','build','proposal'],default='all');a=p.parse_args()
    if a.phase=='all':
        q=load('qkv_build_queue',HERE.parent/'run-nightly.py').Queue(BATCH.name,ROOT,'tle_yy')
        q.run('build-qkv-host',PY+' '+str(__file__)+' --phase build',1200);proposal()
    elif a.phase=='build':build()
    else:proposal()


if __name__=='__main__':main()
