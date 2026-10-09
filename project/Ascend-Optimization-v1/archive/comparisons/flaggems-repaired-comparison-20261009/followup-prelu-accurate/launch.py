import json, os, pathlib, socket, subprocess, time, urllib.request
root = pathlib.Path(__file__).resolve().parent
previous = root.parent / 'followup-prelu-staged'
previous_pid = int((previous/'server.pid').read_text())
print('WAIT_PREVIOUS_BATCH', flush=True)
while True:
    if (previous/'exit-code.txt').exists():
        try: os.kill(previous_pid, 0)
        except ProcessLookupError: break
    time.sleep(5)
print('PREVIOUS_BATCH_FINISHED', flush=True)
pid='3066804'
env = {x.split('=',1)[0]:x.split('=',1)[1] for x in (pathlib.Path('/proc')/pid/'environ').read_bytes().decode().split('\x00') if '=' in x}
env['ASCEND_RT_VISIBLE_DEVICES']='7'
env['KGS_FLAGGEMS_ROOT']='/data/hanle/ascend-optimization/FlagGems'
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(('127.0.0.1',19656))
(root/'npu-before.txt').write_bytes(subprocess.check_output(['npu-smi','info']))
server_log=open(root/'server.log','ab',buffering=0)
server=subprocess.Popen(['/usr/local/python3.11.15/bin/python3.11','-c','from kernelgen_server.server import main; main()',
    '--host','127.0.0.1','--backend','npu','--timing','walltime','--max-workers','1','--port','19656',
    '--profile-artifact-root',str(root/'profiles')],env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'server.pid').write_text(str(server.pid))
try:
    for attempt in range(120):
        if server.poll() is not None: raise RuntimeError('dedicated KGS exited')
        try:
            status=json.load(urllib.request.urlopen('http://127.0.0.1:19656/status',timeout=2))
            if status.get('devices'): break
        except Exception: time.sleep(2)
    else: raise TimeoutError('dedicated KGS startup')
    rc=subprocess.call(['/usr/local/python3.11.15/bin/python3.11','-u',str(root/'tools/compare-repaired-flaggems-910b.py'),
        '--inputs',str(root/'inputs'),'--output',str(root/'measurements'),'--repeats','3','--operators','_prelu_kernel_backward'],env=env)
    (root/'exit-code.txt').write_text(str(rc))
finally:
    server.terminate()
    try: server.wait(timeout=30)
    except subprocess.TimeoutExpired: server.kill(); server.wait()
    (root/'npu-after.txt').write_bytes(subprocess.check_output(['npu-smi','info']))
