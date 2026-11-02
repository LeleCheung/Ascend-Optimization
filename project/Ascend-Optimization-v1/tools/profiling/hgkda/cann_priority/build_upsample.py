"""构建NCHW存储描述的直接CANN插值参考，不执行设备任务。"""
import argparse
import os
from pathlib import Path
import shutil
from common import BATCH,HERE,ROOT,PY,read,save,sha,load


def build():
    record=BATCH/'upsample-reference-build-v2.json'
    if record.exists():
        for p,digest in read(record)['files'].items():assert sha(p)==digest
        return
    os.environ['MAX_JOBS']='1'
    import torch,torch_npu
    from torch.utils.cpp_extension import load as compile_host
    folder=BATCH/'references/upsample_bilinear2d/nchw-v2';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'upsample_v2.cpp';source.write_bytes((HERE/'upsample_v2.cpp').read_bytes())
    npu=Path(torch_npu.__file__).parent
    build_dir=ROOT/'runtime/cann-priority-upsample-nchw-v2';build_dir.mkdir(parents=True,exist_ok=True)
    m=compile_host(name='_hgkda_upsample_nchw_v2',sources=[str(source)],build_directory=str(build_dir),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
            '-L/usr/local/Ascend/cann-9.0.0/lib64','-lopapi','-lascendcl','-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'],
        with_cuda=False,verbose=True)
    binary=folder/Path(m.__file__).name;shutil.copyfile(m.__file__,binary)
    save(record,dict(status='built',binary=str(binary),binary_sha256=sha(binary),source=str(source),source_sha256=sha(source),
        files={str(p):sha(p) for p in [source,binary]},torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,headers={}))


def reference():
    record=BATCH/'upsample-reference-build-v2.json';info=read(record)
    for p,digest in info['files'].items():assert sha(p)==digest
    folder=Path(info['source']).parent;source=folder/'main.py'
    code='import torch,torch_npu,hashlib,importlib.util\nfrom pathlib import Path\n'
    code+='p=Path('+repr(info['binary'])+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(info['binary_sha256'])+'\n'
    code+='s=importlib.util.spec_from_file_location("_hgkda_upsample_nchw_v2",p)\n_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
    code+='''def run(input,output_size,align_corners=False,scales_h=None,scales_w=None):
    return _cpp.run(input,list(output_size),align_corners,0. if scales_h is None else scales_h,0. if scales_w is None else scales_w)
'''
    if source.exists():assert source.read_text(encoding='utf8')==code
    else:source.write_text(code,encoding='utf8')
    save(folder/'provenance.json',dict(source=str(source),source_sha256=sha(source),kind='CANN_layout_compatibility',
        build=str(record),build_sha256=sha(record),scope='直接aclnnUpsampleBilinear2d，NCHW存储描述；非连续输入转连续的成本计入参考'))
    print('UPSAMPLE_REFERENCE',sha(source))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','build','reference'],default='all');a=p.parse_args()
    if a.phase=='all':
        q=load('upsample_build_queue',HERE.parent/'run-nightly.py').Queue(BATCH.name,ROOT,'tle_yy')
        q.run('build-upsample-nchw',PY+' '+str(__file__)+' --phase build',1200);reference()
    elif a.phase=='build':build()
    else:reference()


if __name__=='__main__':main()
