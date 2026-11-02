"""单独构建存储及随机参考，已有二进制不覆盖。"""
import os
from pathlib import Path
from common import BATCH,HERE,ROOT,read,save,sha


def main():
    os.environ['MAX_JOBS']='1'
    import torch,torch_npu
    from torch.utils.cpp_extension import load
    record=BATCH/'cann-build-storage-rng-v1.json'
    if record.exists():
        old=read(record);assert sha(old['source'])==old['source_sha256'] and sha(old['binary'])==old['binary_sha256'];return
    npu=Path(torch_npu.__file__).parent;source=HERE/'cann_storage_rng.cpp'
    build=ROOT/'runtime/cann-priority-storage-rng-20261102-v1';build.mkdir(parents=True,exist_ok=True)
    module=load(name='_hgkda_cann_storage_rng_v1',sources=[str(source)],build_directory=str(build),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
          '-L/usr/local/Ascend/cann-9.0.0/lib64','-lopapi','-lascendcl','-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'],with_cuda=False,verbose=True)
    p=BATCH/'references/artifacts'/Path(module.__file__).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(module.__file__).read_bytes())
    save(record,dict(status='built',module_name='_hgkda_cann_storage_rng_v1',source=str(source),source_sha256=sha(source),
         binary=str(p),binary_sha256=sha(p),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,headers={}))
    print('BUILT_STORAGE_RNG',p)


if __name__=='__main__':main()
