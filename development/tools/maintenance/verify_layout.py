"""核验三部分迁移的冻结原件、交付源码与运行边界。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=a.repository.resolve()
 if os.name=='nt':r=Path(chr(92)*2+'?'+chr(92)+str(r))
 mapping=json.loads((r/'development/manifests/three-part-migration-20261111.json').read_bytes())
 for item in mapping['immutable_files']:assert sha(r/item['new'])==item['sha256'],item['new']
 index=json.loads((r/'operators/index.json').read_bytes());assert index['accepted']==26
 count=0;missing=[]
 def check(v):
  nonlocal count
  if isinstance(v,list):
   for item in v:check(item)
  elif isinstance(v,dict):
   if 'path' in v and 'sha256' in v:
    if v.get('location')=='historical_unavailable':missing.append(v['path']);return
    path=r/v['path'];assert sha(path)==v['sha256'],path;count+=1
   for value in v.values():check(value)
 for item in index['rows']:
  manifest=json.loads((r/'operators'/item['manifest']).read_bytes());check(manifest)
  if item['accepted']:
   assert manifest['selected']['accepted']
   for spec in manifest['checks'].values():
    assert (r/'operators'/item['operator']/spec['checker']).is_file()
    assert (r/'operators'/item['operator']/spec['reference']/'oracle.py').is_file()
 public_files=list((r/'hgkda/src').rglob('*.py'))
 for path in public_files:
  tree=ast.parse(path.read_text(encoding='utf8'))
  for node in ast.walk(tree):
   if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('development'),path
   if isinstance(node,ast.Import):assert not any(x.name.startswith('development') for x in node.names),path
   if isinstance(node,ast.Constant) and isinstance(node.value,str):
    assert 'development/tools/' not in node.value and 'development/experiments/' not in node.value,path
 result=dict(status='passed',immutable_files=len(mapping['immutable_files']),delivery_artifacts=count,
  operators=len(index['rows']),main_total=30,accepted=26,public_python_files=len(public_files),historically_unavailable=missing,
  note='静态边界核验与隔离执行验证分别记录；历史缺失原件不伪造。')
 a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__':main()
