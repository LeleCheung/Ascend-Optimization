"""单CPU检查当前SDK与Triton数学入口，冻结声明和实现线索，不使用NPU。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import triton
    roots=[Path('/usr/local/Ascend/cann-9.0.0/include/aclnnop'),
        Path('/usr/local/Ascend/cann-9.0.0/opp/built-in/op_proto/inc')]
    rows=[];seen=set()
    for root in roots:
        for path in sorted(root.rglob('*.h')):
            f=path.resolve()
            if f in seen:continue
            seen.add(f)
            if not re.search(r'fractional|inverse|(?:^|_)lu(?:_|\.)|linalg.*(?:lu|inv|solve)|uniform|expm1',f.name,re.I):continue
            raw=f.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            out=a.output/'headers'/digest/f.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
            rows.append(dict(path=str(path),resolved=str(f),sha256=digest,snapshot=str(out),
                text=raw.decode(errors='replace')[:24000]))
    tree=Path(triton.__file__).parent
    math=[]
    for f in sorted(tree.rglob('*.py')):
        if 'language' not in f.parts:continue
        raw=f.read_bytes();text=raw.decode(errors='replace')
        matches=[dict(line=i+1,text=v) for i,v in enumerate(text.splitlines())
            if re.search(r'expm1|def exp\(|exp_v|vexp|libdevice',v,re.I)]
        if not matches:continue
        digest=hashlib.sha256(raw).hexdigest();target=a.output/'triton'/f.relative_to(tree)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        math.append(dict(path=str(f),sha256=digest,snapshot=str(target),matches=matches))
    v=dict(status='cpu_inventory_complete',device_executed=False,qualified_CANN=False,
        triton=triton.__version__,sdk_roots=[str(p) for p in roots],headers=rows,triton_math=math,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='仅安装SDK头文件和Triton源码线索，不将接口声明当作运行支持，不执行设备，不修改环境。')
    (a.output/'result.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=v['status'],headers=len(rows),triton_math_files=len(math))))


if __name__=='__main__':main()
