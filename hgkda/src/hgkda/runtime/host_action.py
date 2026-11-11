"""执行方法迁移中的单项宿主操作；只改调度元数据，不写入候选源码。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('action',choices=['generate','preflight','evaluate','pair'])
    p.add_argument('--invocation');p.add_argument('--max-turns',type=int,default=24)
    p.add_argument('--port',type=int,default=19696);p.add_argument('--case')
    a=p.parse_args();c=read(a.config);run=Path(c['run_directory']);package=Path(c['package'])
    workspace=run/'workspace';source=workspace/'candidate.py'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    tools=Path(__file__).parent
    if a.action=='generate':
        assert a.invocation
        c['active_source']='candidate.py' if source.exists() else 'starter.py'
        a.config.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        command=[sys.executable,str(tools/'agent_runner.py'),'--config',str(a.config),
            '--model-settings',str(run/'model-settings.json'),'--invocation',a.invocation,'--max-turns',str(a.max_turns)]
    else:
        digest=sha(source)
        review=run/'source-reviews'/(digest+'.json')
        assert review.exists(),'缺少当前源码的只读审查记录'
        if a.action=='preflight':
            command=[sys.executable,str(tools/'agent_tools.py'),'--config',str(a.config),'preflight','--source',str(source)]
        elif a.action=='evaluate':
            command=[sys.executable,str(tools/'evaluate.py'),'--config',str(a.config),
                '--source',str(source),'--review',str(review),'--port',str(a.port)]
        else:
            assert a.case
            command=[sys.executable,str(tools/'collect_pair.py'),'--config',str(a.config),
                '--source',str(source),'--case',a.case,'--output',str(run/('paired-'+digest[:12]+'-'+a.case))]
    subprocess.run(command,env=env,check=True)
    if a.action=='evaluate':
        formal=run/('formal-'+digest[:12]);feedback=formal/'performance-feedback.json'
        assert read(feedback)['result']['source_sha256']==digest
        shutil.copyfile(feedback,workspace/'performance-feedback.json')
        c['active_source']='candidate.py';a.config.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
