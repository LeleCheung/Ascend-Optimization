"""只读核对安装版本的Ascend C Expm1入口，跟随SDK符号链接并冻结命中头文件。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'清点结果已冻结，不覆盖'
    roots=[args.root/'aarch64-linux/ascendc/include',args.root/'aarch64-linux/asc/include']
    visited=set();headers=0;rows=[];frozen=args.output.with_suffix('')
    for root in roots:
        for current,dirs,files in os.walk(root,followlinks=True):
            resolved=Path(current).resolve()
            if resolved in visited:
                dirs[:]=[]
                continue
            visited.add(resolved)
            for name in sorted(files):
                if not name.endswith(('.h','.hpp','.hxx')):continue
                path=(Path(current)/name).resolve()
                try:raw=path.read_bytes()
                except OSError:continue
                headers+=1;text=raw.decode('utf8',errors='replace')
                if 'expm1' not in text.lower():continue
                digest=hashlib.sha256(raw).hexdigest()
                snapshot=frozen/digest/path.name;snapshot.parent.mkdir(parents=True,exist_ok=True)
                if snapshot.exists():assert snapshot.read_bytes()==raw
                else:snapshot.write_bytes(raw)
                lines=text.splitlines()
                matches=[dict(line=i+1,text=value) for i,value in enumerate(lines) if 'expm1' in value.lower()]
                definitions=[]
                for match in re.finditer(r'inline\s+float\s+(?:Expm1Impl|expm1f)\s*\([^)]*\)\s*\{[^}]*\}',text):
                    definitions.append(match.group(0))
                rows.append(dict(path=str(path),sha256=digest,frozen=str(snapshot),matches=matches,
                    direct_definitions=definitions,
                    architecture_guards=[value for value in lines if '__NPU_ARCH__' in value],
                    explicit_backend_includes=[value for value in lines if '#include' in value and 'dav_c' in value]))
    value=dict(status='cpu_readonly_math_inventory',roots=[str(p) for p in roots],
        resolved_directories=len(visited),headers_scanned=headers,matches=rows,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device_executed=False,qualified_CANN=False,performance_measured=False,
        scope='当前SDK头文件声明及函数体，非设备调用轨迹，不证明910B支持所列SIMT接口；没有改写参考或候选。')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],headers_scanned=headers,matched_headers=len(rows),
                         device_executed=False,qualified_CANN=False)))


if __name__=='__main__':main()
