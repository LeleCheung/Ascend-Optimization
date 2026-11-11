from pathlib import Path
import ast,json,subprocess,hashlib,shutil,os,tarfile
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111'
repo=w/'FlagGems-filtered';a=w/'agent-v2';project=r/'Ascend-Optimization'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
expected='d6a8eec473517a3d68157b208eb9c057eb1d4c50'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected
old=ast.parse((a/'candidate.py').read_text());new=ast.parse((a/'log10.py').read_text())
old_defs={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
new_defs={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
assert all(new_defs.get(k)==v for k,v in old_defs.items())
assert set(new_defs)-set(old_defs)=={'log10_'}
def strip(tree):
 items=[]
 for n in tree.body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
  if isinstance(n,ast.FunctionDef) and n.name=='log10_':continue
  if isinstance(n,ast.Import) and [v.name for v in n.names]==['logging']:continue
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='logger' for t in n.targets):continue
  items.append(n)
 return ast.dump(ast.Module(body=items,type_ignores=[]),include_attributes=False)
assert strip(old)==strip(new),'existing executable code changed'
init='src/flag_gems/runtime/backend/_ascend/ops/__init__.py';module='src/flag_gems/runtime/backend/_ascend/ops/log10.py'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1'],text=True).strip()
assert sha(repo/init)==sha(project/'development/dependencies/third_party/FlagGems'/init)
for src,dest in [('backend_init.py',init),('log10.py',module)]:shutil.copyfile(a/src,repo/dest)
save(w/'integration-review.json',dict(status='reviewed',delivery_sha256=sha(a/'candidate.py'),
 module_sha256=sha(repo/module),all_original_function_asts_preserved=True,original_function_count=len(old_defs),
 original_executable_ast_preserved=True,added_function='log10_',
 candidate_writer='CC/DeepSeek',host_contribution='workflow, repository integration verification, test adapter',
 model_boundary='generation reached turn limit after files and diagnosis were written; outputs independently reviewed'))
method=w/'method-v2';method.mkdir()
with tarfile.open(w/'upstream-method-v2.tar.gz') as t:t.extractall(method,filter='data')
python=r/'runtime/hgkda-release-venv/bin/python'
def run(label,args):
 with (w/(label+'.log')).open('wb') as log:p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
 print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+' failed; inspect log')
run('build-wheel',[python,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',w/'wheels',method/'hgkda'])
run('host-environment',[python,'-m','venv',w/'host-venv'])
host=w/'host-venv/bin/python';wheel=next((w/'wheels').glob('*.whl'))
run('install',[host,'-m','pip','install','--no-deps',wheel])
source=project/'operators/log10_/versions/independent_strict/retained/kernel.py'
config=dict(repository=str(repo),upstream_commit=expected,operator='log10_',
 integration_files={name:sha(repo/name) for name in [init,module]},expected_function_source=module,
 delivery_source=str(source),delivery_sha256=sha(source),
 python=str(r/'runtime/flaggems-validation-venv/bin/python'),sdk='/usr/local/Ascend/cann-9.0.0',
 device=7,lock=str(r/'runs/physical-card-7-campaign.lock'),output=str(w/'attempt-1'),
 container='tle_yy',timeout_seconds=3600)
save(w/'config.json',config)
run('native-tests',[host,'-m','hgkda.runtime','upstream-test','--config',w/'config.json'])
