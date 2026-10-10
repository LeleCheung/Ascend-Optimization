#!/usr/bin/env bash
# 本轮新副本与专用服务：修复 N/A 解析，不改变算子、测试范围或设备计时计算。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
reference_pid=${2:?REFERENCE_KGS_PID}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
port=19657
exec 8>"$r/narrow-device-campaign.lock"
flock -n 8 || exit 2
for session in kg-narrow-timing-v2-20261010 kg-matmul-compiler-v7-20261010; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 15; done
done
"$py" - "$r" "$reference_pid" "$port" <<'PY'
import json,pathlib,socket,subprocess,sys,urllib.request
r=pathlib.Path(sys.argv[1]);pid=int(sys.argv[2]);port=int(sys.argv[3])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for old in (19655,19656):
    d=json.load(opener.open(f'http://127.0.0.1:{old}/status',timeout=15))
    assert d['scheduler']['active']==0 and d['scheduler']['waiting']==0
with socket.socket() as sock:
    assert sock.connect_ex(('127.0.0.1',port)) != 0, '端口已占用，拒绝覆盖'
proc=pathlib.Path('/proc')/str(pid)
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'kernelgen_server.server' in cmd and '--port 19656' in cmd
allow={'PATH','LD_LIBRARY_PATH','LIBRARY_PATH','PYTHONPATH','ASCEND_HOME_PATH',
       'ASCEND_OPP_PATH','ASCEND_TOOLKIT_HOME','CANN_HOME_PATH'}
env={}
for raw in (proc/'environ').read_bytes().split(b'\0'):
    if not raw or b'=' not in raw:continue
    key,value=raw.decode().split('=',1)
    if key in allow:env[key]=value
env.update(ASCEND_RT_VISIBLE_DEVICES='7',KGS_FLAGGEMS_ROOT=str(r/'FlagGems-narrow-device-timing'))
args=['docker','exec','-d']
for key,value in env.items():args+=['-e',key+'='+value]
log=r/f'kgs-{port}.log';profiles=r/'kgs-profiles-narrow-device'
for path in (r,log,profiles):
    assert path.is_absolute() and not any(ch in str(path) for ch in " '\";$`\\\n")
args+=['tle_yy','bash','-c',f'exec /usr/local/python3.11.15/bin/python -u -m kernelgen_server.server --host 127.0.0.1 --backend npu --timing walltime --max-workers 1 --port {port} --profile-artifact-root {profiles} >{log} 2>&1']
subprocess.run(args,check=True)
print('NARROW_DEVICE_KGS_STARTED',port,flush=True)
PY
for ((i=0;i<60;i++)); do
 curl --noproxy '*' -fsS "http://127.0.0.1:$port/status" >/dev/null 2>&1 && break
 sleep 2
done
server=http://127.0.0.1:$port
seed=$p/operators/narrow_copy/candidates/flaggems-master-20261010.py
baseline=$p/operators/narrow_copy/reports/master-device-adapter-20261010
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy --source "$seed" \
 --server "$server" --output "$baseline" --label flaggems-master --timeout 1800 --timing-scope device_task
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --server "$server" --output "$p/operators/narrow_copy/reports/contiguous-copy-device-20261010" \
 --label contiguous-copy --timeout 1800 --timing-scope device_task
export KG_RUN_ROOT="$r/kg-runs" KG_SOURCE_ROOT="$r/Ascend-Optimization/sources"
export KG_WALLTIME_PORT=$port KG_PROFILE_PORT=$port
for arm in no-profile profile; do
 bash "$p/tools/evaluation/run-kg-operator-910b.sh" narrow_copy "$arm" "$arm-device-master-20261010" 2 "$seed" \
  >"$r/campaign-logs/narrow-device-$arm.log" 2>&1 || echo "NARROW_KG_FAILED_$arm"
 "$py" "$p/tools/evaluation/finalize-kg-arm.py" \
  "$r/kg-runs/narrow_copy-$arm-device-master-20261010" narrow_copy \
  "$p/operators/narrow_copy/reports/kg-$arm-device-master-20261010" \
  --label "kg-$arm" --server "$server" --timing-scope device_task \
  || echo "NARROW_KG_INDEPENDENT_FAILED_$arm"
done
"$py" "$p/tools/evaluation/compare-operator-versions.py" \
 --version master "$baseline/flaggems-master-1.result.json" \
 --version kg-no-profile "$p/operators/narrow_copy/reports/kg-no-profile-device-master-20261010/independent/kg-no-profile-1.result.json" \
 --version kg-profile "$p/operators/narrow_copy/reports/kg-profile-device-master-20261010/independent/kg-profile-1.result.json" \
 --version optimized "$p/operators/narrow_copy/reports/contiguous-copy-device-20261010/contiguous-copy-1.result.json" \
 --output "$p/operators/narrow_copy/reports/master-closure-20261010/版本对比.md"
echo 'NARROW_DEVICE_FOUR_VERSIONS_PASSED'
