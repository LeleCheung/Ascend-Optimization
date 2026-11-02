"""只读清点剩余算子的CANN入口及旧ACL图算子定义，不推导运行支持或性能。"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'已有清点冻结保留'
    headers=[]
    for p in sorted((args.root/'include/aclnnop').rglob('*.h')):
        if not re.search(r'fractional|inverse|expm1|uniform|(?:^|_)lu(?:_|\.)',p.name,re.I):continue
        text=p.read_text(errors='replace')
        declarations=re.findall(r'\baclnnStatus\s+(aclnn\w+)\s*\([^;]+;',text,re.S)
        headers.append(dict(path=str(p),sha256=sha(p),functions=declarations))
    graph=[]
    for p in sorted((args.root/'opp/built-in/op_proto/inc').glob('*.h')):
        text=p.read_text(errors='replace')
        for match in re.finditer(r'REG_OP\(\s*([^\)]+)\)',text):
            name=match.group(1).strip()
            if not re.search(r'fractional|inverse|^lu$',name,re.I):continue
            end=text.find('OP_END_FACTORY_REG',match.end())
            end=text.find('\n',end) if end>=0 else match.end()+1200
            start=text.rfind('/**',0,match.start())
            start=max(match.start()-3500,start if start>=0 else match.start())
            graph.append(dict(name=name,path=str(p),sha256=sha(p),
                              declaration_and_context=text[start:end][:6500]))
    lib=ctypes.CDLL('libopapi.so')
    names=['aclnnForeachExpm1','aclnnForeachExpm1GetWorkspaceSize',
           'aclnnFractionalMaxPool3d','aclnnFractionalMaxPool3DWithFixedKsize',
           'aclnnLinalgInvEx','aclnnLinalgLuFactorEx','aclnnInverse','aclnnInplaceUniform']
    exports={name:bool(getattr(lib,name,None)) for name in names}
    value=dict(status='cpu_inventory_complete',root=str(args.root),headers=headers,graph_ops=graph,exports=exports,
        checked_roots=[str(args.root/'include/aclnnop'),str(args.root/'opp/built-in/op_proto/inc')],
        checker_sha256=sha(__file__),device_executed=False,qualified_CANN=False,
        scope='检查安装版本头文件、声明、旧ACL图算子和指定导出名；接口存在不等于支持本合同，未发现也不证明其它版本/所有组合都不存在。')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],headers=len(headers),graph_names=sorted({x['name'] for x in graph}),exports=exports)))


if __name__=='__main__':main()
