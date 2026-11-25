"""Compatibility for early SELU recovery contracts; use transfer_context for new experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from hgkda.artifacts import unwrap
from hgkda.context import knowledge


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_initial(config):
    """Replay only the early SELU context format; do not prepare candidates or extend to other operators."""
    assert config['job']['operator']=='selu_', 'Legacy context supports SELU only; use operator_transfer configuration for new experiments'
    run=Path(config['run_directory']);workspace=run/'workspace'
    comparison=workspace/'comparison.json';context=workspace/'context.json'
    if not comparison.exists():
        subprocess.run([sys.executable,'-m','hgkda','compare','--directory',config['paired_directory'],
                        '--output',str(comparison)],check=True,stdout=subprocess.DEVNULL)
    if not context.exists():
        subprocess.run([sys.executable,'-m','hgkda','context','--operator','selu_',
            '--source',str(workspace/'starter.py'),'--comparison',str(comparison),
            '--ir-manifest',str(run/'baseline-ir/manifest.json'),'--output',str(context)],
            check=True,stdout=subprocess.DEVNULL)
    data=unwrap(read(context));paired=data['facts']['paired']
    local_ir=workspace/'actual-ir';local_ir.mkdir(exist_ok=True)
    for item in data['facts']['compiled_ir']:
        shutil.copyfile(item['path'],local_ir/Path(item['path']).name)
    selected=[row for row in paired['differences'] if any(k in row['metric'] for k in
                ['ratio','time','instructions','datas'])]
    selected.sort(key=lambda row:abs((row.get('candidate_over_reference') or 1)-1),reverse=True)
    brief=dict(source_sha256=data['source']['sha256'],case=data['case'],reference_kind=data['reference_kind'],
        differences=selected[:12],transfer_granularity=paired['derived_differences'],hypotheses=data['hypotheses'],
        mechanisms=data['knowledge']['mechanisms'],full_context=str(context),
        measurement_limits=paired['limitations'],
        interpretation_rules=['A zero capture does not establish sub-resolution cost or no overhead; preserve the unknown cause.',
            'Tile payload is not actual per-instruction movement granularity; verify compiled output or counters.',
            'Do not divide full-task bytes by per-block average time to establish Roofline or claim bandwidth saturation.',
            'Verify precision changes in the new candidate IR; Python expressions do not replace compilation evidence.'],
        actual_ir=[str(local_ir/Path(row['path']).name) for row in data['facts']['compiled_ir']],
        reference='Native CANN; formal performance comes only from full KGS evaluation')
    (workspace/'context-brief.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return brief


def build(config):
    run=Path(config['run_directory']);workspace=run/'workspace'
    source=workspace/'starter.py';digest=sha(source)
    probe=Path(config['probe_directory']);measured=read(probe/'probe.json')
    feedback=unwrap(read(workspace/'performance-feedback.json'))
    assert measured['source_sha256']==feedback['source_sha256']==digest
    actual_ir=workspace/'actual-ir';actual_ir.mkdir(exist_ok=True)
    artifacts=[]
    for index,entry in enumerate(measured['compiled']):
        for item in entry['artifacts']:
            path=Path(item['path']);assert sha(path)==item['sha256']
            if path.suffix=='.npubin':continue
            target=actual_ir/(str(index)+'-'+path.name);shutil.copyfile(path,target)
            artifacts.append(dict(path=str(target),sha256=item['sha256'],kernel=entry['kernel']))
    # List device tasks separately; exclude duplicated derived intervals such as Computing.
    trace_root=Path(config['trace_directory']);trace_tasks=[]
    assert read(trace_root/'probe.json')['source_sha256']==digest, 'Task trace belongs to a different source'
    for item in read(trace_root/'traces.json'):
        path=Path(item['path']);assert sha(path)==item['sha256']
        document=read(path)
        events=document['traceEvents'] if isinstance(document,dict) else document
        tasks=[e for e in events if e.get('ph')=='X' and e.get('args',{}).get('Task Type')]
        assert tasks, 'Trace lacks vendor device-task records'
        trace_tasks.append(dict(case=item['case'],side=item['side'],trace_sha256=item['sha256'],
            kernel_names=sorted({e['name'] for e in tasks}),count=len(tasks),
            durations_us=[e['dur'] for e in tasks],median_us=statistics.median(e['dur'] for e in tasks)))
    library=knowledge('selu_',hypotheses=True)
    brief=dict(operator='selu_',source_sha256=digest,formal_feedback=feedback,
        independent_launch_probe=measured['measurements'],device_tasks=trace_tasks,
        launchers=[{k:v for k,v in entry.items() if k not in ['artifacts','metadata']} for entry in measured['compiled'][:2]],
        mechanisms=library['mechanisms'],knowledge_retrieval=library['retrieval'],actual_ir=artifacts,
        limitations=['Asynchronous submission includes host logic and possible backpressure; wall-submit does not establish device time.',
            'Device tasks come from independent profiling windows and cannot replace full-call KGS scores.',
            'Do not reuse old-source PipeUtilization/Roofline or invent missing component metrics.',
            'Historical launch optimization and current diagnosis suggest methods; validate combined benefits by evaluating new source fully.'])
    (workspace/'context-brief.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return brief
