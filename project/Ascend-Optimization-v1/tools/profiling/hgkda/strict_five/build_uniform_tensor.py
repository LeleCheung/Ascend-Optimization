"""单CPU构建厂商UniformTensor参考入口；不初始化NPU，不改共享环境。"""
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
OUT=PROJECT/'operators/campaigns/strict-five-20261103/reference-probes/uniform-tensor-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['MAX_JOBS']='1';os.environ['OMP_NUM_THREADS']='1'
    import torch
    import torch_npu
    def forbidden(*args,**kwargs):raise RuntimeError('构建禁止NPU初始化')
    torch.npu._lazy_init=forbidden
    from torch.utils.cpp_extension import load
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'build.json').exists()
    build=ROOT/'runtime/strict-five-20261103/uniform-tensor-v1';assert not build.exists();build.mkdir(parents=True)
    cann=Path('/usr/local/Ascend/cann-9.0.0');npu=Path(torch_npu.__file__).parent;source=HERE/'cann_uniform_tensor.cpp'
    name='_hgkda_uniform_tensor_v1'
    module=load(name=name,sources=[str(source)],build_directory=str(build),with_cuda=False,verbose=True,
        extra_include_paths=[str(npu/'include'),str(cann/'include')],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
            '-L'+str(cann/'lib64'),'-lopapi','-lascendcl','-Wl,-rpath,'+str(cann/'lib64')])
    binary=OUT/Path(module.__file__).name;binary.write_bytes(Path(module.__file__).read_bytes())
    frozen=OUT/source.name;frozen.write_bytes(source.read_bytes());header=cann/'include/aclnnop/aclnn_uniform.h'
    target=OUT/header.name;target.write_bytes(header.read_bytes())
    value=dict(status='built',module_name=name,binary=str(binary),binary_sha256=sha(binary),source=str(frozen),
        source_sha256=sha(source),header=str(target),header_sha256=sha(header),builder_sha256=sha(__file__),
        device_executed=False,qualified_CANN=False,performance_measured=False,torch=torch.__version__,
        scope='只封装现有原生UniformTensor接口，无边界修改、输出裁剪或自写数值修正；未作设备或性能评测。')
    (OUT/'build.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='built',binary_sha256=sha(binary))))


if __name__=='__main__':main()
