"""统计实际捕获TTIR的运算种类；IR操作数不冒充硬件指令计数。"""
import hashlib
import json
import re
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[4]
    batch=project/'operators/campaigns/threeway-20261030'
    rows=[]
    for manifest in sorted((batch/'actual-ir/cross_out').glob('*/manifest.json')):
        value=json.loads(manifest.read_text(encoding='utf8'))
        for entry in value['entries']:
            for artifact in entry['artifacts']:
                if artifact['kind']!='ttir':continue
                path=Path(artifact['path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
                counts={op:sum(bool(re.match(r'^\s*(?:%[^=]+?=\s*)?'+re.escape(op)+r'\s',line)) for line in raw.decode('utf8').splitlines())
                        for op in ['arith.remsi','tt.gather','arith.fptosi','tt.load','tt.store']}
                rows.append(dict(source=value['spec']['source'],source_sha256=value['spec']['source_sha256'],
                    case=value['spec']['case']['name'],ttir=str(path),ttir_sha256=artifact['sha256'],operations=counts))
    result=dict(status='observed',operator='cross_out',rows=rows,
        conclusion='当前FP32大输入实际TTIR：起点的remsi和四次gather在移位读取候选中被消除；分块1024变为4096。',
        scope='源码绑定的实际编译IR结构；不是模拟器耗时、硬件指令计数或分离单项贡献的因果实验')
    (batch/'actual-ir/cross_out/ir-facts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
