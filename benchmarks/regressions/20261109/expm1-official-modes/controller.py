"""等待已排队实验结束，再用预先创建的私有缓存重试官方精度模式探测。"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_bytes())


def read(path):return json.loads(Path(path).read_bytes())


def save(value):
    path=ROOT/'controller-state.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


state=dict(status='waiting_for_preceding_stages',reason='GE/TBE私有缓存目录缺失的环境修复')
save(state)
try:
    previous=Path(CONFIG['preceding_run'])
    while True:
        receipt=previous/'controller-state.json'
        if receipt.exists() and read(receipt)['status'] in ['completed','requires_attention']:
            if (previous/'state.json').exists():
                assert read(previous/'state.json')['status']=='completed'
                assert (previous/'service-stop.json').exists()
            break
        time.sleep(10)
    with Path(CONFIG['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert hashlib.sha256((ROOT/'probe.py').read_bytes()).hexdigest()==CONFIG['probe_sha256']
        for mode in ['default','high_precision']:
            state.update(status='probing_official_reference',mode=mode);save(state)
            command=[CONFIG['device_python'],str(ROOT/'probe.py'),'--mode',mode,'--sdk',CONFIG['sdk'],
                     '--catalog',CONFIG['catalog'],'--output',str(ROOT/mode),'--physical-device',str(CONFIG['device'])]
            shell='source '+shlex.quote(str(Path(CONFIG['sdk'])/'set_env.sh'))+'\n'
            shell+='export ASCEND_RT_VISIBLE_DEVICES='+shlex.quote(str(CONFIG['device']))+'\n'
            shell+='export PYTHONDONTWRITEBYTECODE=1\nexport OMP_NUM_THREADS=1\nexec '+shlex.join(command)
            with (ROOT/(mode+'.log')).open('xb') as output:
                result=subprocess.run(['docker','exec',CONFIG['container'],'bash','-c',shell],stdout=output,stderr=subprocess.STDOUT)
            assert result.returncode==0
            assert (ROOT/mode/'result.json').exists()
        state.update(status='completed',results={mode:read(ROOT/mode/'result.json') for mode in ['default','high_precision']})
        save(state)
except Exception as error:
    state.update(status='requires_attention',error=type(error).__name__+': '+str(error));save(state)
    raise
