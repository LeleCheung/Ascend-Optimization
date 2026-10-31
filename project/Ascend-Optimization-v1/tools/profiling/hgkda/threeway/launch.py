"""在910B宿主独立tmux启动指定批次；只创建本项目会话。"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',default='threeway-20261030')
    p.add_argument('--phase',default='verify',choices=['verify','references','search','proposals','finalize','all'])
    p.add_argument('--operators',nargs='*')
    a=p.parse_args()
    if not a.batch.replace('-','').isalnum():raise ValueError('非法批次')
    session='hanle-'+a.batch+'-'+a.phase
    if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('已有同名tmux，不重复启动')
    runtime=ROOT/'runs'/a.batch;runtime.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'runtime/kg-controller/venv/bin/python'),str(PROJECT/'tools/profiling/hgkda/threeway/run.py'),'--batch',a.batch,'--phase',a.phase]
    if a.operators:command+=['--operators',*a.operators]
    log=runtime/(a.phase+'-controller.log')
    body=shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,body],check=True)
    print('STARTED '+session+' '+str(log))


if __name__=='__main__':main()
