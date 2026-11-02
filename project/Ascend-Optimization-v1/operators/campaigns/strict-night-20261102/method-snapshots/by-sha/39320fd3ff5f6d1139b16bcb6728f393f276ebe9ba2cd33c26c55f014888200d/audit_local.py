"""在Windows或Linux核对交付快照和冻结审计的全部已保存文件身份。"""
import hashlib
import json
import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
REMOTE='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH=PROJECT/'operators/campaigns/strict-night-20261102'


def main():
    checked={}
    def check(name,digest):
        assert str(name).startswith(REMOTE+'/'),'证据必须在项目内：'+str(name)
        path=PROJECT/str(name)[len(REMOTE)+1:]
        path=path.resolve();path.relative_to(PROJECT)
        if os.name=='nt':path=Path('\\\\?\\'+str(path))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, str(path)
        checked[str(name)]=digest
    def load(path):return json.loads(path.read_text(encoding='utf8'))
    delivery=load(BATCH/'delivery.json')
    for name,digest in delivery['evidence_sha256'].items():check(name,digest)
    frozen=BATCH.parent/'strict-triton-20261102'
    for file,field in [('delivery.json','evidence_sha256'),('delivery-audit.json','files_sha256'),
                       ('reference-qualification-audit.json','evidence_sha256')]:
        for name,digest in load(frozen/file).get(field,{}).items():check(name,digest)
    assert len(delivery['rows'])==30 and sum(row['accepted'] for row in delivery['rows'])==delivery['accepted']
    result=dict(status='verified',operators=30,accepted=delivery['accepted'],files=len(checked),
                delivery_sha256=hashlib.sha256((BATCH/'delivery.json').read_bytes()).hexdigest(),
                scope='本机交付快照＋冻结审计的文件实体与SHA核验；不触发设备测试，不把审计通过解释为30项性能均达标')
    (BATCH/'local-copy-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
