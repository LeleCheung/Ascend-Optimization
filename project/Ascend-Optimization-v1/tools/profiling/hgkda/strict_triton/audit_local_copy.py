"""离线核对当前仓库中的原始证据SHA；映射910B绝对路径，不执行设备工作。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

PREFIX='/data/hanle/ascend-optimization/Ascend-Optimization/'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.repo.resolve();batch=root/'project/Ascend-Optimization-v1/operators/campaigns/strict-triton-20261102'
    checked={}
    def local(name):
        path=root/name[len(PREFIX):] if name.startswith(PREFIX) else root/name
        path=path.resolve();path.relative_to(root)
        return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path
    def read(path):return json.loads(path.read_text(encoding='utf8'))
    def check(name,digest):
        path=local(name);actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==digest,(name,digest,actual)
        checked[name]=digest
    delivery=read(batch/'delivery.json');audit=read(batch/'delivery-audit.json');assert audit['status']=='verified'
    check((batch/'delivery.json').relative_to(root).as_posix(),audit['delivery_sha256'])
    for value,key in [(delivery,'evidence_sha256'),(audit,'files_sha256'),(read(batch/'reference-qualification-audit.json'),'evidence_sha256')]:
        for name,digest in value[key].items():check(name,digest)
    dependency=read(batch/'dependency-review.json')
    for name,v in dependency['files'].items():check(name,v['sha256']);check(v['snapshot'],v['sha256'])
    for row in delivery['rows']:
        check(row['source'],row['source_sha256']);source=local(row['source']);review=read(source.parent/'review.json')
        assert review['source_sha256']==row['source_sha256'] and review['all_branches_reviewed'] and review['dependencies_reviewed']
        for bridge in review.get('reviewed_host_bridges',[]):
            check(bridge['binary'],bridge['binary_sha256']);check(bridge['binary_snapshot'],bridge['binary_sha256']);check(bridge['cpp_source'],bridge['cpp_source_sha256'])
        proof=read(local(row['compiled_path']['path']));assert proof['status']=='verified' and proof['source_sha256']==row['source_sha256']
        for name,digest in proof['artifacts_sha256'].items():check(name,digest)
    assert len(delivery['rows'])==30 and sum(r['accepted'] for r in delivery['rows'])==delivery['accepted']
    result=dict(status='verified',operators=30,accepted=delivery['accepted'],files=len(checked),
        evidence_sha256=checked,scope='离线核验本地实体字节与本轮审计记录；不替代设备测量，不表示所有算子性能达标')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
    print(json.dumps(dict(status=result['status'],operators=30,accepted=result['accepted'],files=result['files'])))


if __name__=='__main__':main()
