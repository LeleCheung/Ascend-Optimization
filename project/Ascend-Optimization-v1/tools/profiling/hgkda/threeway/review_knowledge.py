"""将源码审查已拒绝的实现退出知识检索，原记录完整保留。"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here.parent/'next_phase'))
    from knowledge import organize
    project=here.parents[3];batch=project/'operators/campaigns/threeway-20261030'
    denied=json.loads((batch/'disqualified-sources.json').read_text(encoding='utf8'))
    path=batch/'knowledge/records.json'
    if not path.exists():return
    raw=path.read_bytes();records=json.loads(raw)
    changed=[]
    for record in records:
        sha=record.get('evidence',{}).get('source_sha256')
        if sha in denied and record.get('status')!='rejected':
            record.update(status='rejected',source_review=denied[sha])
            changed.append(record.get('id'))
    if not changed:return
    backup=path.with_name('records-before-review-'+hashlib.sha256(raw).hexdigest()[:12]+'.json')
    if backup.exists() and backup.read_bytes()!=raw:raise FileExistsError(backup)
    backup.write_bytes(raw)
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    index=organize(records)
    (path.parent/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (path.parent/'source-review-update.json').write_text(json.dumps(dict(status='reviewed',rejected=changed,
        previous=str(backup),previous_sha256=hashlib.sha256(raw).hexdigest(),
        reason='已验证的完整成绩不能覆盖新发现的源码语义错误；拒绝实现退出成功知识检索'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='reviewed',rejected=len(changed))))


if __name__=='__main__':main()
