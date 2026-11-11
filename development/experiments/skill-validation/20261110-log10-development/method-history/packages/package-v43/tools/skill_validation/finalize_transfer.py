"""结束方法迁移试验，保存方法实体与全部正式结果；不改写候选或原始测量。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from audit_generation import audit


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def copy_verified_tree(source,target):
    """失败后续做归档时保留已复制原件，只接受完全相同的树。"""
    files=lambda root:{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts}
    if target.exists():assert files(source)==files(target),'已复制方法树与来源不同'
    else:shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--outcome',choices=['accepted','failed'],required=True)
    p.add_argument('--failure-reason')
    p.add_argument('--method-history-root',type=Path)
    p.add_argument('--retained-delivery-score',type=float)
    a=p.parse_args();run=a.directory;config=read(run/'config.json')
    assert not (run/'summary.json').exists()
    calls=list((run/'agent-calls').iterdir())
    assert calls and all((path/'model-usage.json').exists() for path in calls),'模型调用仍在运行或原件不完整'
    records=[]
    for directory in sorted(run.glob('formal-*')):
        if not directory.is_dir():continue
        assert read(directory/'state.json')['status']=='completed',directory
        assert (directory/'service-stop.json').exists(),directory
        document=read(directory/'audit-final.json');assert document['evidence_verified']
        row=document['rows'][0];records.append(dict(directory=directory.name,**row))
    accepted=[row for row in records if row['accepted']]
    if a.outcome=='accepted':assert accepted
    else:assert a.failure_reason and not accepted
    package=Path(config['package'])
    copy_verified_tree(package,run/'method-snapshot')
    if a.method_history_root:
        assert config['phase']=='development'
        target=run/'method-history/packages';target.mkdir(parents=True,exist_ok=True)
        for path in sorted(a.method_history_root.glob('package-v*')):
            if path.is_dir():copy_verified_tree(path,target/path.name)
    generation=audit(run)
    save(run/'generation-audit.json',generation)
    chosen=max(accepted,key=lambda row:row['geo_mean']) if accepted else None
    current=read(Path(config['repository'])/'benchmarks/current.json')
    value=dict(schema_version='hgkda.method-transfer/v1',date=datetime.now(timezone.utc).date().isoformat(),operator=config['operator'],
        phase=config['phase'],outcome=a.outcome,accepted=bool(chosen),actual_generated=bool(chosen),
        source_sha256=chosen['source_sha256'] if chosen else None,
        cann_score=chosen['geo_mean'] if chosen else None,
        formal_directory=chosen['directory'] if chosen else None,
        formal_cases=chosen['formal_cases'] if chosen else config['formal_cases'],
        full_passed=chosen['full_passed'] if chosen else False,
        supplementary_each=config['job']['checks']['supplementary']['previous_total'],
        formal_requests=len(records),formal_history=records,
        preflight_sources=len(list(run.glob('attempts/*/preflight.json'))),model_calls=len(calls),
        prior_best_source_provided=False,codex_direct_candidate_edits=0,
        method_manifest_sha256=sha(run/'method-manifest.json'),generation_audit_sha256=sha(run/'generation-audit.json'),
        retained_existing_delivery_cann_score=a.retained_delivery_score,
        current_original_contract_accepted=current['accepted'],target_total=current['total'],
        attribution=dict(candidate='CC／DeepSeek生成及修改全部候选代码',
            methods='Codex分析、固化workflow／知识／工具并完成只读审查和评测调度'),
        formal_requests_policy='每个新源码SHA一次完整请求，固定合同与计时；不按同源码复测挑高分')
    if config['phase']=='frozen':value['temporary_codex_optimization_prompts']=0
    if a.failure_reason:value['failure_reason']=a.failure_reason
    save(run/'summary.json',value)
    state=read(run/'controller-state.json');state.update(status='completed',outcome=a.outcome,
        completed_at=datetime.now(timezone.utc).isoformat(),summary='summary.json')
    save(run/'controller-state.json',state)
    print(json.dumps({k:value[k] for k in ['phase','outcome','cann_score','formal_requests','model_calls']}))


if __name__=='__main__':main()
