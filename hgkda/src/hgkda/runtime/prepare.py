"""Prepare a run exclusively from a public package, delivery and explicit environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from hgkda.adapters.bundle import pack

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['operator-directory','package','environment','run-directory']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();operator=a.operator_directory.resolve();package=a.package.resolve();run=a.run_directory.resolve()
 m=read(operator/'manifest.json');selected=m['selected'];assert selected and selected['accepted'],'delivery has no accepted selected version'
 # Manifest paths are relative to the distribution root, not to a development repository.
 distribution=operator.parent.parent;source=distribution/selected['path'];assert sha(source)==selected['sha256']
 env=read(a.environment);required=['container','device','device_python','sdk','lock','kgs_server']
 assert all(k in env for k in required),'explicit device/KGS environment required'
 run.mkdir(parents=True,exist_ok=False);workspace=run/'workspace';workspace.mkdir()
 snapshot=run/'method-package';snapshot.mkdir()
 for name in ['src','skills']:
  shutil.copytree(package/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(snapshot/'skills/hgkda',workspace/'skills/hgkda')
 methods=[dict(path=x.relative_to(snapshot).as_posix(),sha256=sha(x)) for x in snapshot.rglob('*') if x.is_file()]
 save(run/'method-manifest.json',dict(files=methods))
 for name in ['starter.py','candidate.py']:(workspace/name).write_bytes(source.read_bytes())
 save(workspace/'starter-provenance.json',dict(kind='existing_delivered_candidate',source_sha256=sha(source),prior_best_source_provided=True))
 reviews=run/'source-reviews';reviews.mkdir();review=distribution/selected['review']['path'];assert sha(review)==selected['review']['sha256'];shutil.copyfile(review,reviews/(sha(source)+'.json'))
 # Relocation metadata is provenance, not an executable KGS contract member.
 bundle,digest=pack(operator/'contracts',exclude={'relocation.json'});(run/'operator.tar').write_bytes(bundle)
 prior=read(operator/'evidence/retained/request.json');prior['binding']['bundle_id']='sha256:'+digest
 prior['implementation']['sources'][0]['content']=source.read_bytes().decode('utf8');save(run/'request.json',prior)
 checks={}
 for name,entry in m['checks'].items():
  leaf=operator/entry['reference'];checker=operator/entry['checker']
  checks[name]=dict(leaf=str(leaf),checker=str(checker),checker_sha256=sha(checker),previous_total=entry['expected_cases'],
    contract_sha256={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']})
 shutil.copytree(operator/'references/cann',run/'reference-catalog')
 c=dict(env,operator=m['operator'],context_kind='operator_transfer',phase='development',active_source='candidate.py',
   package=str(snapshot),run_directory=str(run),method_manifest=str(run/'method-manifest.json'),require_preflight_review=True,
   formal_cases=m['formal_cases'],maximum_preflight_sources=3,maximum_formal_sources=3,
   job=dict(operator=m['operator'],archive=str(run/'operator.tar'),request=str(run/'request.json'),checks=checks,
            reference_kind=m['reference_kind']))
 c['special_checks']={}
 for name,spec in m.get('special_checks',{}).items():
  checker=operator/spec['checker'];entry=dict(special=True,checker=str(checker),checker_sha256=sha(checker),previous_total=spec['expected_cases'])
  if spec.get('reference'):entry['leaf']=str(operator/spec['reference'])
  c['special_checks'][name]=entry
 save(run/'config.json',c);save(workspace/'contract.json',dict(operator=m['operator'],formal_cases=m['formal_cases'],
   supplementary_each=checks['supplementary']['previous_total'],definition=read(operator/'contracts/definition.json')))
 save(run/'controller-state.json',dict(status='prepared'))
 print(json.dumps(dict(status='prepared',config=str(run/'config.json'),source_sha256=sha(source),prior_best_source_provided=True)))
if __name__=='__main__':main()
