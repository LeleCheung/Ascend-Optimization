"""等待主评测正常结束，顺序补充厂商参考并继续低分项搜索。"""
from pathlib import Path
import subprocess
import time
from common import BATCH, HERE, ROOT, read, save


def main():
    runtime=ROOT/'runs'/BATCH.name
    marker=runtime/'followup-state.json'
    while subprocess.run(['tmux','has-session','-t','hanle-cann-priority-20261102-all'],capture_output=True).returncode==0:
        save(marker,dict(status='waiting_for_main_queue',dependency='hanle-cann-priority-20261102-all'))
        time.sleep(10)
    state=read(BATCH/'next-state.json')
    if state.get('status')!='phase_completed' or state.get('current_job') is not None:
        raise RuntimeError('主队列尚未正常结束，不绕过未决任务')
    host=ROOT/'runtime/kg-controller/venv/bin/python'
    for phase in ['extra','search']:
        save(marker,dict(status='running',phase=phase))
        subprocess.run([str(host),'-u',str(HERE/'run.py'),'--phase',phase],check=True)
    save(marker,dict(status='completed',phases=['extra','search']))


if __name__=='__main__':main()
