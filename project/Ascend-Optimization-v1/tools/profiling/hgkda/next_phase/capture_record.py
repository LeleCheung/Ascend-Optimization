"""对已完整验证的指定记录补采同输入证据，不重新搜索或重发正式评测。"""
import hashlib
import json
import math
from pathlib import Path


def choose_case(cases):
    if not cases:raise ValueError('没有冻结的性能输入')
    preferred=[case for case in cases if case.get('inputs',{}).get('spec',{}).get('dtype')=='float32']
    return max(preferred or cases,key=lambda case:math.prod(case.get('inputs',{}).get('spec',{}).get('shape',[])))


def capture(r, record, *, update_latest=False):
    from report import verify
    from run import PY,HERE,quote,load,save_json
    checked=verify(record)
    if (checked.get('status')!='verified' or not checked.get('correctness_passed') or
            checked.get('extra_status')!='passed'):
        raise ValueError('指定记录尚未通过当前完整评测和补测，不能作为已验证源补采')
    supplementary=load(Path(record['extra_checks']));op=supplementary['operator']
    leaf=r.batch/r.q.records[op]['catalog_leaf']
    contracts=r.q.records[op]['contract_sha256']
    for name,expected in contracts.items():
        if hashlib.sha256((leaf/name).read_bytes()).hexdigest()!=expected:
            raise r.module.Attention('补采输入合同改变：'+name)
    cases=[json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    case=choose_case(cases);case_id=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:12]
    source=Path(record['source']);sha=record['source_sha256'];tag=sha[:12]+'-'+case_id
    analysis=r.batch/'analysis'/op
    spec=analysis/('spec-selected-'+tag+'.json');out=analysis/('paired-selected-'+tag)
    payload=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=sha,case=case,
                 contract_sha256=contracts,scope='已完整验证记录的冻结性能输入；优先FP32最大输入，不缩小形状')
    if spec.exists() and load(spec)!=payload:raise r.module.Attention('已有补采规格不同，拒绝覆盖')
    if not spec.exists():save_json(spec,payload)
    if not out.exists():
        r.job('paired-selected:'+op+':'+tag,lambda:r.q.run('paired-selected-'+op+'-'+tag,
            quote([PY,HERE/'collect.py','--spec',spec,'--output',out]),2400))
    if not (out/'state.json').exists():
        return dict(status='capture_unavailable',directory=str(out),source_sha256=sha)
    context=analysis/('context-selected-'+tag+'.json')
    if not context.exists():
        r.job('context-selected:'+op+':'+tag,lambda:r.q.run('context-selected-'+op+'-'+tag,
            quote([PY,HERE/'assemble.py','--capture',out,'--output',context]),600))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
        raise r.module.Attention('补采过程中源码改变，保留原件，不登记最新证据')
    if context.exists() and update_latest:
        save_json(analysis/'latest-context.json',dict(path=str(context),sha256=hashlib.sha256(context.read_bytes()).hexdigest()))
    return dict(status=load(out/'state.json')['status'],directory=str(out),context=str(context),source_sha256=sha)
