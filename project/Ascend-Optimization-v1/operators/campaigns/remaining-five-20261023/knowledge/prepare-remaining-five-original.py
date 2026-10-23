#!/usr/bin/env python3
"""建立剩余五算子的独立合同与连续任务，不改历史原件。"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parents[2]
REPO = PROJECT.parents[1]
ORDER = ['upsample_bilinear2d', '_conj_physical', 'scatter_add', 'cumulative_trapezoid', 'fractional_max_pool3d']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())

def prepare(output):
    if output.exists(): raise FileExistsError(output)
    previous = PROJECT/'operators/campaigns/night-20261021-v3'
    old = json.loads((previous/'plan.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('five_contract_helper', TOOLS/'prepare-nightly.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    rows=[]
    for op in ORDER:
        origin_op = 'conj_physical' if op == '_conj_physical' else op
        row = dict(next(x for x in old['records'] if x['operator'] == origin_op))
        leaf = output/'catalog/ops'/op; leaf.mkdir(parents=True)
        for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
            shutil.copyfile(previous/row['catalog_leaf']/name, leaf/name)
        seed = output/'seeds'/(op+'.py'); seed.parent.mkdir(exist_ok=True)
        source = previous/row['seed']
        if op == 'scatter_add':
            source = PROJECT/'operators/交付源码-20261023/scatter_add/main.py'
        elif op == '_conj_physical':
            source = REPO/'third_party/FlagGems/src/flag_gems/ops/_conj_physical.py'
            text=source.read_text(encoding='utf-8')+'\nrun = _conj_physical\n'
            seed.write_bytes(text.encode())
            definition=json.loads((leaf/'definition.json').read_text(encoding='utf-8'))
            definition['name']=op
            definition['effects']={'mutates':[],'returns_alias_of':{}}
            save(leaf/'definition.json',definition)
            oracle=(leaf/'oracle.py').read_text(encoding='utf-8')
            # 输入构建沿用旧公共接口的固定配方，仅参考调用更正为私有接口。
            assert oracle.count("return torch.conj_physical(kwargs['input'])")==1
            oracle=oracle.replace("return torch.conj_physical(kwargs['input'])", "return torch.ops.aten._conj_physical(kwargs['input'])")
            (leaf/'oracle.py').write_bytes(oracle.encode())
            for name in ['correctness.jsonl','timing.jsonl']:
                (leaf/name).write_bytes((leaf/name).read_bytes().replace(b'conj_physical-',b'_conj_physical-'))
            row.update(operator=op, excel_name=op, upstream_source=source.relative_to(REPO).as_posix(),
                upstream_source_sha256=sha(source), coverage='正确私有接口；沿用旧输入集合，补充独立别名、布局及性能输入数值门禁')
        if op != '_conj_physical': shutil.copyfile(source,seed)
        row.update(seed=seed.relative_to(output).as_posix(), seed_origin=source.relative_to(REPO).as_posix(),
            source_sha256=sha(seed), catalog_leaf=leaf.relative_to(output).as_posix(),
            contract_sha256={p.name:sha(p) for p in leaf.iterdir()}, source='excel_meeting')
        rows.append(row)
    save(output/'catalog/manifest.json',dict(api_version='v6.2',name=output.name,evaluator='native',layout='per-operator'))
    knowledge=output/'knowledge'; knowledge.mkdir()
    shutil.copyfile(TOOLS/'remaining-five-brief.md',knowledge/'夜间任务.md')
    refs=TOOLS/'skills/hgkda/references'
    for name in ['算子类型/归约与扫描.md','算子类型/索引与采样.md','性能指标/成对对照与机制定位.md']:
        shutil.copyfile(refs/name,knowledge/Path(name).name)
    tools=['prepare-remaining-five.py','run-remaining-five.py','check-remaining-five.py',
        'run-endurance.py','check-endurance.py','observe-endurance.py','run-nightly.py','build-kg-overlay.py']
    save(knowledge/'tool-sha256.json',{n:sha(TOOLS/n) for n in tools})
    plan=dict(schema_version='hgkda-remaining-five-v1',physical_card=7,server='http://127.0.0.1:19659',
        timing_scope='walltime',final_repeats=1,duration_hours=12,
        base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),
        passes=[dict(rounds=4,budget_seconds=3000),dict(rounds=3,budget_seconds=2400)],profile_budget_seconds=240,
        records=rows,policy='五项持续串行；单项失败后继续；原始PyTorch含CPU回退必须标注，不能静默换reference')
    save(output/'plan.json',plan)
    (output/'README.md').write_bytes(('''# 会议剩余五算子达标批次

目标是五项正确性修复与性能达标。使用固定FlagGems/FlagTree、物理卡7、KGS19659完整调用walltime。各候选一次完整正式复验及补充语义、全部性能输入数值检查。服务器tmux连续运行，失败保留后转下一项；运行状态不等于最终交付。

对象：upsample_bilinear2d、_conj_physical、scatter_add、cumulative_trapezoid、fractional_max_pool3d。_conj_physical新建正确接口合同；scatter从上一批0.419×有效源码继续，其余从固定上游起步。合同和源码SHA见plan.json。

CPU回退参考与原生NPU参考分别标注，不通过调整容差或替换参考制造达标。最终源码在deliverables/，过程与失败见endurance-state.json、results/和research/。目标、方法预算与贡献见knowledge/夜间任务.md。
''').encode())
    return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(dict(batch=args.output.name,operators=len(prepare(args.output)['records']))))
