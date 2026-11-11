from pathlib import Path
import ast,json,subprocess,hashlib,shutil,os,tarfile
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111'
repo=w/'FlagGems-filtered';project=r/'Ascend-Optimization'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def run(label,args):
 with (w/(label+'.log')).open('wb') as log:p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
 print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+' failed; inspect log')
method=w/'method-v3';method.mkdir()
with tarfile.open(w/'upstream-method-v3.tar.gz') as t:t.extractall(method,filter='data')
python=r/'runtime/hgkda-release-venv/bin/python';host=w/'host-venv/bin/python'
run('build-wheel-v3',[python,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',w/'wheels-v3',method/'hgkda'])
wheel=next((w/'wheels-v3').glob('*.whl'));run('install-v3',[host,'-m','pip','install','--no-deps','--force-reinstall',wheel])
module='src/flag_gems/runtime/backend/_ascend/ops/log10.py'
source=w/'agent-profiler-fix/log10.py';shutil.copyfile(source,repo/module)
review=dict(operator='log10_',source=str(source),source_sha256=sha(source),status='reviewed_strict_triton',
 all_branches_reviewed=True,dependencies_reviewed=True,reviewer='Codex source and runtime-contract review',
 candidate_writer='CC/DeepSeek',origin_sha256=sha(w/'agent-v2/log10.py'),
 review_note='Only direct-launch tensor argument changed from integer address to Tensor. Device kernel bodies, layout branches, stream lookup, alignment decision, fallback, empty handling and cache keys retained. Logging is host metadata only. No native compute path.')
save(w/'review-profiler-fixed.json',review)
c=json.loads((w/'config.json').read_bytes());c.update(output=str(w/'attempt-2'),mode='operator-files',
 accuracy_files=['tests/test_log10.py'],benchmark_files=['benchmark/test_log10.py'],accuracy_reference='cpu',
 expected_accuracy_cases_from=str(w/'attempt-1/upstream/log10_/accuracy_result.json'),
 source_review=str(w/'review-profiler-fixed.json'))
c['integration_files'][module]=sha(source);save(w/'config-fixed.json',c)
run('native-tests-fixed',[host,'-m','hgkda.runtime','upstream-test','--config',w/'config-fixed.json'])
