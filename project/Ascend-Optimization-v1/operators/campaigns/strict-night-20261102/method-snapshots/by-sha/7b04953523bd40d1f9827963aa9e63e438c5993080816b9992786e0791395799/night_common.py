"""本批共享路径和文件操作；避免与其他队列的run模块重名。"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-night-20261102'
PY='/usr/local/python3.11.15/bin/python3.11'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)
