"""单CPU构建原生CANN ForeachExpm1参考适配；不初始化或使用NPU执行算子。"""
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ['MAX_JOBS']='1'
import torch
import torch_npu
from torch.utils.cpp_extension import load

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/strict-night-20261102'
ROOT=PROJECT.parents[2]
OUT=BATCH/'reference-probes/native-foreach-expm1-v1'
BUILD=ROOT/'runtime/strict-night-20261102/reference-foreach-expm1-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    record=OUT/'build.json'
    if record.exists():
        old=json.loads(record.read_text())
        assert old['source_sha256']==sha(HERE/'cann_foreach_expm1.cpp')
        assert old['binary_sha256']==sha(old['binary'])
        print(json.dumps(dict(status='reused',record=str(record))))
        return
    BUILD.mkdir(parents=True,exist_ok=True)
    npu=Path(torch_npu.__file__).parent
    cann=Path('/usr/local/Ascend/cann-9.0.0')
    source=HERE/'cann_foreach_expm1.cpp'
    name='_hgkda_foreach_expm1_v1'
    module=load(name=name,sources=[str(source)],build_directory=str(BUILD),
        extra_include_paths=[str(npu/'include'),str(npu/'include/third_party/op-plugin'),str(cann/'include')],
        extra_cflags=['-O2','-g0'],with_cuda=False,verbose=True,
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
                       '-L'+str(cann/'lib64'),'-lopapi','-lascendcl','-Wl,-rpath,'+str(cann/'lib64')])
    binary=OUT/Path(module.__file__).name
    binary.write_bytes(Path(module.__file__).read_bytes())
    headers=[cann/'include/aclnnop/level2/aclnn_foreach_expm1.h',
             npu/'include/third_party/op-plugin/op_plugin/utils/op_api_common_base.h']
    dependencies={}
    for header in headers:
        frozen=OUT/header.name;frozen.write_bytes(header.read_bytes())
        dependencies[str(header)]=dict(sha256=sha(header),frozen=str(frozen))
    frozen=OUT/source.name;frozen.write_bytes(source.read_bytes())
    value=dict(status='built',module_name=name,source=str(frozen),source_sha256=sha(source),
        binary=str(binary),binary_sha256=sha(binary),dependencies=dependencies,
        torch=torch.__version__,torch_npu=torch_npu.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        architecture=platform.machine(),builder_sha256=sha(__file__),
        scope='单CPU构建；直接CANN ForeachExpm1与官方tensor-list元数据转换，未进行NPU数值或性能评测，不是候选')
    record.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='built',record=str(record),binary_sha256=sha(binary))))


if __name__=='__main__':main()
