"""只编译宿主C++包装，不分配NPU张量；产物放在仓库外runtime。"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1'
    import torch
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('host_bridge.cpp')
    source_bytes=source.read_bytes()
    library=load(name='_hgkda_host_bridge_v1',sources=[str(source)],
        build_directory=str(args.output),extra_cflags=['-O2','-g0'],with_cuda=False,verbose=True)
    assert source.read_bytes()==source_bytes,'构建过程中源码改变，拒绝登记产物身份'
    binary=Path(library.__file__)
    result=dict(status='built',source=str(source),source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        python=sys.version,torch=torch.__version__,architecture=platform.machine(),
        abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        scope='只编译通用宿主元数据和启动包装；没有执行目标算子或分配NPU张量')
    (args.output/'build-provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
