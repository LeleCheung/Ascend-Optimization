"""仅重采已确认语义且唯一符号的漏选侧；保留旧采集和失败状态。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

from collect import run, save, HERE


def repair(capture):
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    original=capture/'state.json'
    state=json.loads(original.read_text(encoding='utf8'))
    backup=capture/'state.before-symbol-repair.json'
    if not backup.exists():backup.write_bytes(original.read_bytes())
    spec=capture/'symbol-repair-spec.json'
    if spec.exists():assert json.loads(spec.read_text(encoding='utf8'))==state['spec']
    else:save(spec,state['spec'])
    for side,row in state['sides'].items():
        if row['status']=='completed':continue
        out=capture/side
        semantic=out/'semantics.json'
        if not semantic.exists() or json.loads(semantic.read_text(encoding='utf8'))['status']!='passed':continue
        names=row.get('selected',[])
        logs=list(out.glob('kernel-*.log'))
        discovered=list(dict.fromkeys(name for log in logs for name in re.findall(r'Kernel (.+?) skipped: not selected via --kernel-name\.',log.read_text(errors='replace'))))
        if len(names)!=1 or len(discovered)!=1:continue
        name=discovered[0];profile=out/'symbol-repair-1'
        if not profile.exists():
            app=shlex.join([sys.executable,str(HERE/'driver.py'),'--spec',str(spec),'--side',side])
            run(['msprof','op','--application='+app,'--kernel-name='+name,'--launch-count=1','--warm-up=0','--aic-metrics=Roofline','--output='+str(profile)],out/'symbol-repair-1.log',timeout=480)
        basics=list(profile.rglob('OpBasicInfo.csv'))
        if len(basics)!=1:raise RuntimeError('符号重采仍缺唯一基本信息 '+side)
        with basics[0].open(encoding='utf-8-sig',newline='') as file:basic=list(csv.DictReader(file))
        assert len(basic)==1 and basic[0]['Op Name']==name
        row.update(status='completed',previous_error=row.pop('error',None),
            kernel_name_binding=dict(trace_names=names,op_names=[name],method='同一目标路径唯一调度任务与msprof唯一漏选符号；旧失败原件保留'),
            captures=[dict(kernel=name,directory=str(profile),basic=basic[0],files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in profile.rglob('*.csv')})])
        save(original,state)
    state['status']='completed' if all(r['status']=='completed' for r in state['sides'].values()) else 'partial'
    state['symbol_repair']=dict(tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),before=str(backup),before_sha256=hashlib.sha256(backup.read_bytes()).hexdigest())
    save(original,state)
    print(json.dumps(dict(status=state['status'],sides={k:v['status'] for k,v in state['sides'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True);repair(parser.parse_args().capture)
