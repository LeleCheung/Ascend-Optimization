#!/usr/bin/env python3
"""冻结夜间连续优化队列；复用已验证合同，不重建或缩减计分用例。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parents[2]
REPO = PROJECT.parents[1]
ORDER = ['selu_', 'rad2deg_', 'cumulative_trapezoid', 'scatter_add', 'uniform', 'gelu_and_mul']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    previous = PROJECT/'operators/campaigns/night-20261021-v3'
    buffer = PROJECT/'operators/campaigns/buffer-pointwise-20261022'
    old = json.loads((previous/'plan.json').read_text(encoding='utf-8'))
    records = []
    for op in ORDER:
        row = dict(next(r for r in old['records'] if r['operator'] == op))
        leaf = output/'catalog/ops'/op
        leaf.mkdir(parents=True)
        origin = previous/row['catalog_leaf']
        for n in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
            shutil.copyfile(origin/n, leaf/n)
        seed = output/'seeds'/(op+'.py')
        seed.parent.mkdir(exist_ok=True)
        if op == 'selu_':
            delivered = json.loads((buffer/'deliverables-final/manifest.json').read_text(encoding='utf-8'))
            chosen = next(r for r in delivered['records'] if r['operator'] == op)
            origin_seed = buffer/'deliverables-final'/op/chosen['selected']['source_sha256']/'main.py'
        else:
            origin_seed = previous/row['seed']
        shutil.copyfile(origin_seed, seed)
        row.update(seed=seed.relative_to(output).as_posix(), source_sha256=sha(seed),
                   seed_origin=origin_seed.relative_to(REPO).as_posix(),
                   catalog_leaf=leaf.relative_to(output).as_posix(),
                   contract_sha256={p.name:sha(p) for p in sorted(leaf.iterdir())})
        records.append(row)
    (output/'catalog/manifest.json').write_text(json.dumps(dict(api_version='v6.2',
        name=output.name, evaluator='native', layout='per-operator'),indent=2)+'\n',encoding='utf-8')
    knowledge = output/'knowledge'
    knowledge.mkdir()
    for name in ['方法与机制.md', 'final-comparison.json', 'completion-audit.json']:
        shutil.copyfile(buffer/name, knowledge/name)
    brief = TOOLS/'endurance-brief.md'
    shutil.copyfile(brief, knowledge/'夜间任务.md')
    tools = ['prepare-endurance.py','run-endurance.py','check-endurance.py','observe-endurance.py',
             'run-nightly.py','build-kg-overlay.py','check-pointwise-semantics.py',
             'check-dense-pointwise.py','check-pointwise-ranges.py']
    (knowledge/'tool-sha256.json').write_text(json.dumps({n:sha(TOOLS/n) for n in tools},indent=2)+'\n',encoding='utf-8')
    plan = dict(schema_version='hgkda-endurance-v1', base_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO).decode().strip(), physical_card=7,
        server='http://127.0.0.1:19659', timing_scope='walltime', final_repeats=1,
        duration_hours=12, passes=[dict(rounds=4,budget_seconds=3600),dict(rounds=3,budget_seconds=2700)],
        profile_budget_seconds=240, records=records,
        policy='各步骤自动衔接；候选失败保留证据并继续，未知设备任务先核对；按实际延迟选版，不按参考波动挑分')
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return plan


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(dict(operators=len(prepare(a.output)['records']),batch=a.output.name)))
