"""逐项核验实际IR是否变化，仅移除调试位置，不推断二进制或性能等价。"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def without_locations(line):
    out=[];index=0;quoted=False;escaped=False
    while index<len(line):
        char=line[index]
        if char=='"' and not escaped:quoted=not quoted
        if not quoted and line.startswith('loc(',index) and (index==0 or not (line[index-1].isalnum() or line[index-1]=='_')):
            while out and out[-1].isspace():out.pop()
            index+=4;depth=1;inner_quoted=False;inner_escape=False
            while index<len(line) and depth:
                current=line[index]
                if current=='"' and not inner_escape:inner_quoted=not inner_quoted
                if not inner_quoted:
                    if current=='(':depth+=1
                    elif current==')':depth-=1
                inner_escape=(current=='\\' and not inner_escape)
                index+=1
            assert depth==0,'调试位置表达式未闭合'
            escaped=False
            continue
        out.append(char);escaped=(char=='\\' and not escaped);index+=1
    return ''.join(out)


def normalized(text):
    lines=[]
    for line in text.splitlines():
        if re.match(r'\s*#loc\w*\s*=',line):continue
        line=without_locations(line)
        lines.append(line.rstrip())
    return '\n'.join(lines).strip()+'\n'


def compare(left,right):
    results=[]
    assert left['cases']==right['cases'],'捕获的case或执行路径不同，不能按位置比较'
    assert len(left['entries'])==len(right['entries']),'实际编译产物数量不同'
    for index,(a,b) in enumerate(zip(left['entries'],right['entries'])):
        assert a['symbol']==b['symbol'],'内核顺序或名称不同'
        for kind in ['ttir','bcmlir']:
            pair=[]
            for entry in [a,b]:
                artifact=next(item for item in entry['artifacts'] if item['kind']==kind)
                raw=Path(artifact['path']).read_bytes()
                assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
                pair.append(normalized(raw.decode('utf8')))
            digests=[hashlib.sha256(text.encode()).hexdigest() for text in pair]
            results.append(dict(index=index,symbol=a['symbol'],kind=kind,equal=pair[0]==pair[1],
                                left_normalized_sha256=digests[0],right_normalized_sha256=digests[1]))
    return dict(left_source_sha256=left['source_sha256'],source_sha256=right['source_sha256'],
        all_normalized_ir_equal=bool(results) and all(row['equal'] for row in results),
        comparisons=results,scope='按同序实际捕获比较TTIR和bcmlir，仅去除#loc定义及位置引用；不视作二进制或性能等价证明')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['left','right','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    result=compare(json.loads(args.left.read_bytes()),json.loads(args.right.read_bytes()))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:result[key] for key in ['left_source_sha256','source_sha256','all_normalized_ir_equal']}))


if __name__=='__main__':main()
