"""持项目卡锁执行排队候选；既有请求不重发，结果保留。"""
import fcntl
import importlib.util
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'remaining'))
spec=importlib.util.spec_from_file_location('remaining_previous_controller',HERE.parent/'remaining/run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def main():
    with (previous.base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=previous.Runner('remaining-final-20261031')
        r.state['active_started_epoch']=time.time();r.save()
        try:
            r.process_proposals();r.wait_idle()
            r.state.update(status='proposals_completed',current_job=None,finished_at=r.module.now());r.save()
        except Exception as e:
            r.state.update(status='needs_attention',error=type(e).__name__+': '+str(e));r.save();raise

if __name__=='__main__':main()
