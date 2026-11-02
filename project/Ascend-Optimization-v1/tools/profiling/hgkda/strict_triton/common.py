"""严格Triton复核：独立批次，不修改历史证据。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-triton-20261102'
CANN=HERE.parent/'cann_priority'
PREVIOUS=PROJECT/'operators/campaigns/cann-priority-20261102'
NEXT=HERE.parent/'next_phase'
PY='/usr/local/python3.11.15/bin/python3.11'
sys.path.insert(0,str(NEXT))

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+chr(10)).encode('utf8'))
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
