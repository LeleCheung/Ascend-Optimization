"""在当前队列释放卡锁后运行短验证；不与已有KG作业同时占卡。"""
import argparse
import fcntl
from run import Runner, ROOT, HERE, PY, quote, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);a=p.parse_args()
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=Runner(a.batch)
        r.wait_idle()
        results={}
        tasks=[('reference-qualification-v1','check_references.py',[]),('input-path-probe-v1','probe_inputs.py',['--batch',r.batch])]
        for key,script,args in tasks:
            output=r.batch/(key+'.json')
            if output.exists():
                results[key]=dict(status='existing',path=str(output));continue
            try:
                r.q.run(key,quote([PY,HERE/script,*args,'--output',output]),600)
                results[key]=dict(status='completed',path=str(output))
            except Exception as error:
                results[key]=dict(status='failed',error=str(error))
            save_json(r.batch/'maintenance-state-v1.json',results)
        save_json(r.batch/'maintenance-completed-v1.json',dict(status='completed',tasks=results,time=r.module.now()))


if __name__=='__main__':main()
