#!/usr/bin/env python3
"""只整理文件：将独立门禁与准确合同打包到KG的tmp目录，供KGS调试任务运行。"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re

TOOLS=Path(__file__).resolve().parent
CONTRACT=['definition.json','oracle.py','correctness.jsonl','timing.jsonl']
OPERATORS={'rad2deg_','cumulative_trapezoid','scatter_add','uniform','gelu_and_mul'}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def stage(batch,workspace,source,output,operator):
    batch=batch.resolve();workspace=workspace.resolve();source=source.resolve();output=output.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_]+',operator) or operator not in OPERATORS:
        raise ValueError('此入口服务check-endurance的五项算子，SELU使用三项专用检查')
    source.relative_to(workspace);output.relative_to((workspace/'tmp').resolve())
    if output.exists():raise FileExistsError('请使用新的tmp打包目录')
    plan_raw=(batch/'plan.json').read_bytes();plan=json.loads(plan_raw)
    record=next(r for r in plan['records'] if r['operator']==operator)
    if set(record['contract_sha256'])!=set(CONTRACT):raise ValueError('合同文件集合不完整')
    leaf=PurePosixPath(record['catalog_leaf'])
    if leaf.is_absolute() or '..' in leaf.parts or '\\' in str(leaf) or ':' in str(leaf):raise ValueError('非法合同路径')
    original=(batch.joinpath(*leaf.parts)).resolve();original.relative_to(batch)
    checker=(TOOLS/'check-endurance.py').read_bytes()
    frozen=json.loads((batch/'knowledge/tool-sha256.json').read_text(encoding='utf-8'))
    if sha(checker)!=frozen['check-endurance.py']:raise ValueError('门禁工具与本批登记身份不同')
    candidate=source.read_bytes();candidate.decode('utf-8')
    payloads={'candidate/main.py':candidate,'check-endurance.py':checker,'plan.json':plan_raw}
    for name in CONTRACT:
        raw=(original/name).read_bytes()
        if sha(raw)!=record['contract_sha256'][name]:raise ValueError('合同SHA改变: '+name)
        payloads[leaf.joinpath(name).as_posix()]=raw
    output.mkdir(parents=True)
    files=[]
    for name,raw in payloads.items():
        target=output.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        files.append(dict(path=target.relative_to(workspace).as_posix(),sha256=sha(raw),size_bytes=len(raw)))
    manifest=dict(schema_version='hgkda-debug-semantic-bundle-v1',operator=operator,
        batch=str(batch),source=str(source),source_sha256=sha(candidate),plan_sha256=sha(plan_raw),
        checker_sha256=sha(checker),contract_sha256=record['contract_sha256'],files=files,
        scope='只复制准确字节；不运行NPU、不修改正式合同、不代替一次正式KGS完整评测')
    manifest_path=output/'bundle-manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prefix=output.relative_to(workspace).as_posix()
    request=dict(purpose='本轮候选独立语义和冻结性能输入验证，不参与正式计时',
        command=['{python}',prefix+'/check-endurance.py','--operator',operator,
            '--source',prefix+'/candidate/main.py','--output','artifacts/independent-'+sha(candidate)[:12]+'.json'],
        files=[dict(path=x['path']) for x in files]+[dict(path=prefix+'/bundle-manifest.json')],timeout_seconds=600)
    result=dict(bundle=str(output),source_sha256=sha(candidate),bundle_manifest_sha256=sha(manifest_path.read_bytes()),debug_request=request)
    (output/'debug-request.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--operator',required=True)
    p.add_argument('--workspace',type=Path,default=Path.cwd());p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(stage(a.batch_dir,a.workspace,a.source,a.output,a.operator),ensure_ascii=False))
