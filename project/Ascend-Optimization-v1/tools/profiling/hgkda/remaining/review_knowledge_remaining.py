"""按源码身份合并新门禁；原生路径失败不能留在成功实现检索中。"""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'
sys.path.insert(0,str(HERE.parent/'next_phase'))
from knowledge import organize


def main():
    path=BATCH/'knowledge/records.json'
    records=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
    state=json.loads((BATCH/'next-state.json').read_text(encoding='utf8'))
    for record in records:
        evidence=record['evidence'];source=Path(evidence['source']);op=record['scope']['operator']
        best=state['operators'].get(op,{}).get('best') or {}
        if (record.get('status')=='validated' and best.get('source_sha256')
                and evidence['source_sha256']!=best['source_sha256']):
            record['status']='stale'
            record['superseded_reason']='本批已有更好的合格实现；时间更新不代表性能更好，较慢记录保留为历史'
            continue
        manifest=source.parent/'manifest.json'
        if not manifest.exists():continue
        value=json.loads(manifest.read_text())
        if value.get('implementation_language')!='python':continue
        record['implementation_backend']=value['backend']
        gate=source.parent/'assessment/native-preflight.json'
        if not gate.exists():
            best=state['operators'].get(op,{}).get('best') or {}
            if best.get('source_sha256')==evidence['source_sha256'] and best.get('native_safety'):
                gate=Path(best['native_safety'])
        if not gate.exists():
            record['status']='draft';record['validation_note']='需要对齐原生路径门禁后才能作为成功实现复用';continue
        checked=json.loads(gate.read_text())
        assert checked['source_sha256']==evidence['source_sha256']
        attachment=dict(path=str(gate),sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
        # 原扩展合同与native路径合同不同，不能把24项检查冒充原完整补测。
        evidence['native_safety']=attachment
        if checked['status']!='passed':
            extras=evidence.setdefault('supplementary',[])
            if attachment not in extras:extras.append(attachment)
            record.update(status='observed_failure',summary='原完整合同通过，但对齐原生路径追加语义检查失败；禁止按成功实现直接复用')
        else:
            result=json.loads(Path(evidence['formal_result']).read_text())
            record['measured_speedup']=result.get('geo_mean')
            record['threshold_passed']=bool(result.get('geo_mean',0)>.8)
    index=organize(records)
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (path.parent/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    summary=dict(status='reviewed',active=len(index['active']),superseded=len(index['superseded']),
        validated=sum(r['status']=='validated' for r in index['active']),
        observed_failure=sum(r['status']=='observed_failure' for r in index['active']),
        draft=sum(r['status']=='draft' for r in index['active']),
        scope='成功实现同时核验原合同与对齐原生路径；未达性能门槛的正确实现不称为达标；失败约束按证据保留')
    (path.parent/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
