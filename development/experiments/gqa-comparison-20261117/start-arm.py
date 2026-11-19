import argparse,json,subprocess,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('arm',choices=['no-simulator','with-simulator','no-simulator-v2','with-simulator-v2']);p.add_argument('round');a=p.parse_args()
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117');run=r/a.arm
c=json.loads((run/'config.json').read_bytes());c.update(paired_directory=str(r/'paired'),formal_cases=4)
c['job']['checks']['supplementary']['previous_total']=4
(run/'config.json').write_text(json.dumps(c,indent=2))
context=json.loads((run/'workspace/context-brief.json').read_bytes());context['experiment_contract']=json.loads((r/'contrast-manifest.json').read_bytes())
(run/'workspace/context-brief.json').write_text(json.dumps(context,ensure_ascii=False,indent=2))
prompt='comparison-prompt-round02.md' if a.round=='round-02' else ('comparison-prompt-v2.md' if a.arm.endswith('-v2') else 'comparison-prompt.md')
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),'--model-settings',str(run/'model-settings.json'),'--invocation',a.round,'--max-turns','36','--prompt-file',str(r/prompt)]
with (run/('generation-'+a.round+'.log')).open('x') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src')),start_new_session=True)
print(child.pid)
