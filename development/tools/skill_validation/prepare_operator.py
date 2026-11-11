"""准备算子迁移试验；复制固定起点原字节，不编写或改动候选实现。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repository','package','regression-run','run-directory','starter-source','lock','sdk']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--operator',required=True)
    parser.add_argument('--phase',choices=['development','frozen'],required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--device-python',required=True)
    parser.add_argument('--maximum-preflight-sources',type=int,default=8)
    args=parser.parse_args()
    repo=args.repository.resolve();package=args.package.resolve();run=args.run_directory.resolve()
    assert args.starter_source.resolve().is_relative_to(repo/'development/dependencies/third_party/FlagGems')
    run.mkdir(parents=True,exist_ok=False);workspace=run/'workspace';workspace.mkdir()
    plan=read(args.regression_run/'plan.json');job=next(r for r in plan['jobs'] if r['operator']==args.operator)
    leaf=run/'reference-catalog';shutil.copytree(job['checks']['cann_supplementary']['leaf'],leaf)
    oracle=leaf/'oracle.py';text=oracle.read_text(encoding='utf8')
    for item in plan['compatibility_files']:
        if item['path'] in text:
            assert sha(item['archived'])==item['sha256']
            text=text.replace(item['path'],item['archived'])
    oracle.write_text(text,encoding='utf8')
    shutil.copyfile(args.starter_source,workspace/'starter.py')
    save(workspace/'starter-provenance.json',dict(upstream=str(args.starter_source),
         upstream_sha256=sha(args.starter_source),starter_sha256=sha(workspace/'starter.py'),
         operation='原字节复制；未添加入口或候选补丁，全部适配由生成模型完成',prior_best_source_provided=False))
    # 当前宿主提供Edit；预备原始字节文件，避免模型为新建文件转向shell写入。
    shutil.copyfile(workspace/'starter.py',workspace/'candidate.py')
    save(workspace/'candidate-initialization.json',dict(kind='upstream_bytes_copy',
        source_sha256=sha(workspace/'starter.py'),candidate_sha256=sha(workspace/'candidate.py'),
        implementation_changed=False,operation='仅原字节复制固定起点，未适配入口或改变实现'))
    catalog={name:[json.loads(line) for line in (leaf/(name+'.jsonl')).read_text().splitlines() if line.strip()]
             for name in ['correctness','timing']}
    save(workspace/'contract.json',dict(operator=args.operator,formal_cases=sum(len(r) for r in catalog.values()),
        supplementary_each=job['checks']['supplementary']['previous_total'],
        timing_cases=catalog['timing'],settings=job['settings'],reference=job['reference_kind'],
        layouts=sorted({str(r.get('inputs',{}).get('spec',{}).get('strided',False)) for r in catalog['correctness']}),
        threshold=.8,public_entrypoint='run(A)',supplementary_requirements='原固定检查器全部通过，包括特殊值、别名、布局、stream和同shape对齐序列'))
    shutil.copytree(package/'skills/hgkda',workspace/'skills/hgkda')
    (workspace/'.claude/skills').mkdir(parents=True)
    (workspace/'.claude/skills/hgkda').symlink_to(workspace/'skills/hgkda',target_is_directory=True)
    method_files=[p for folder in ['src/hgkda','skills/hgkda']
                  for p in (package/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    method_manifest=run/'method-manifest.json'
    save(method_manifest,dict(phase=args.phase,files=[dict(path=p.relative_to(package).as_posix(),sha256=sha(p))
         for p in sorted(method_files)],policy='调用前核验方法身份；模型仅修改workspace候选与诊断'))
    config=dict(repository=str(repo),package=str(package),run_directory=str(run),operator=args.operator,
        context_kind='operator_transfer',goal_kind='method_transfer_with_formal_feedback',phase=args.phase,
        method_manifest=str(method_manifest),active_source='candidate.py',
        formal_cases=sum(len(r) for r in catalog.values()),job=job,lock=str(args.lock),sdk=str(args.sdk),
        container=args.container,device=args.device,device_python=args.device_python,
        agent_tool_path=str(package/'src/hgkda/runtime/agent_tools.py'),
        maximum_preflight_sources=args.maximum_preflight_sources,maximum_formal_sources=4,
        require_preflight_review=True,threshold=.8)
    save(run/'config.json',config)
    save(run/'controller-state.json',dict(status='prepared',started_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='prepared',operator=args.operator,phase=args.phase,
                         run_directory=str(run),starter_sha256=sha(workspace/'starter.py'))))


if __name__=='__main__':main()
