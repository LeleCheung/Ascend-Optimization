"""在匹配的 CANN/PyTorch NPU 环境重建示例参考宿主桥。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cann-root',type=Path,required=True)
    parser.add_argument('--native-directory',type=Path,required=True)
    parser.add_argument('--build-directory',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    import platform
    os.environ['MAX_JOBS']='1'
    source=args.native_directory/'cann_baselines.cpp'
    args.build_directory.mkdir(parents=True,exist_ok=False)
    npu=Path(torch_npu.__file__).parent
    module=load(name='_hgkda_cann_priority_v1',sources=[str(source.resolve())],
        build_directory=str(args.build_directory.resolve()),extra_include_paths=[str(npu/'include'),str(args.cann_root/'include')],
        extra_cflags=['-O2','-g0'],extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
          '-L'+str(args.cann_root/'lib64'),'-lopapi','-lascendcl','-Wl,-rpath,'+str(args.cann_root/'lib64')],
        with_cuda=False,verbose=True)
    binary=Path(module.__file__)
    output=args.build_directory/'reference'
    output.mkdir()
    (output/binary.name).write_bytes(binary.read_bytes())
    (output/source.name).write_bytes(source.read_bytes())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(module_name='_hgkda_cann_priority_v1',source=source.name,source_sha256=sha(source),binary=binary.name,
                  binary_sha256=sha(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                  architecture=platform.machine(),reference_api='aclnnInplaceMuls',candidate_backend='Triton')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'built','reference_directory':str(output),
                      'next':'将新参考目录用于新合同快照，并重新打包请求；不覆盖旧证据'}))


if __name__=='__main__':main()
