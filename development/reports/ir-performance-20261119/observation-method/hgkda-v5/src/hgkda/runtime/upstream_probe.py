"""核验设备进程加载的 FlagGems 目录和目标函数，不执行目标算子计算。"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--operator',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import flag_gems
    root=a.repository.resolve()
    package=Path(flag_gems.__file__).resolve()
    function=inspect.unwrap(getattr(flag_gems,a.operator))
    module=sys.modules[function.__module__]
    path=Path(inspect.getsourcefile(function)).resolve()
    assert package.is_relative_to(root) and path.is_relative_to(root),'加载到了其他FlagGems安装'
    value=dict(package=str(package),operator=a.operator,module=function.__module__,
               source=path.relative_to(root).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               vendor=flag_gems.vendor_name,device=flag_gems.device,
               kgs_same_entrypoint=(getattr(module,'run',None) is getattr(module,a.operator,None)
                                    if hasattr(module,'run') else None))
    with a.output.open('x',encoding='utf8') as out:json.dump(value,out,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
