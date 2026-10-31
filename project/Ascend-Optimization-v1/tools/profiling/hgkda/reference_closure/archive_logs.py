"""快照本批控制日志，保留崩溃原件；不读取Claude全局设置或凭据。"""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'

def main():
    runtime=ROOT/'runs'/BATCH.name
    candidates=list(runtime.glob('*.log'))+[ROOT/'runs/reference-closure-controller.log',ROOT/'runs/reference-closure-finish.log']
    records=[]
    for source in candidates:
        if not source.exists():continue
        raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        target=BATCH/'controller-logs'/(source.stem+'-'+digest[:16]+'.log')
        target.parent.mkdir(exist_ok=True)
        if target.exists():assert target.read_bytes()==raw
        else:target.write_bytes(raw)
        records.append(dict(source=str(source),snapshot=str(target),sha256=digest,bytes=len(raw),scope='本批日志某次完整字节快照；活动任务可能继续追加'))
    (BATCH/'controller-log-index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    print(json.dumps(dict(logs=len(records))))

if __name__=='__main__':main()
