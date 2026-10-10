#!/usr/bin/env bash
# 仅在本轮其他卡7任务全部结束后启动 narrow 专用 KGS，旧服务保留。
set -euo pipefail
r=${1:?usage: run-master-campaign-910b.sh EXPERIMENT_ROOT REFERENCE_KGS_PID}
reference_pid=${2:?usage: run-master-campaign-910b.sh EXPERIMENT_ROOT REFERENCE_KGS_PID}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
port=19656
mkdir -p "$r/campaign-logs"
exec 8>"$r/narrow-campaign.lock"
flock -n 8 || { echo '已有 narrow campaign'; exit 2; }

sessions=(kg-master-continue-v2-20261010 kg-master-no-profile-recovery-20261010
          kg-v2-and-amin-v4-20261010 kg-amin-v4-continuation-20261010
          kg-matmul-v3-eval-20261010 kg-finalize-master-20261010
          kg-roofline-and-global-20261010 kg-matmul-tuning-20261010
          kg-amin-grid-master-20261010 kg-matmul-v4-eval-20261010
          kg-amin-direct-tiles-20261010 kg-matmul-grouped-v6-20261010
          kg-amin-bf16-v6-20261010)
while true; do
  running=0
  for session in "${sessions[@]}"; do
    tmux has-session -t "$session" 2>/dev/null && running=1
  done
  test "$running" -eq 0 && break
  sleep 15
done

# 再检查当前 master 服务队列。端口与卡号固定，拒绝覆盖或重启已有服务。
"$py" - "$r" "$reference_pid" "$port" <<'PY'
import json,os,pathlib,socket,subprocess,sys,urllib.request
r=pathlib.Path(sys.argv[1]); pid=int(sys.argv[2]); port=int(sys.argv[3])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
x=json.load(opener.open('http://127.0.0.1:19655/status',timeout=30))
if x['scheduler']['active'] or x['scheduler']['waiting']:
    raise SystemExit('master KGS 尚有任务，不能启动同卡 narrow 测试')
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1',port)) == 0:
        raise SystemExit('19656 已占用，拒绝修改现有服务')
proc=pathlib.Path('/proc')/str(pid)
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
if 'kernelgen_server.server' not in cmd or '--port 19655' not in cmd:
    raise SystemExit('参考 KGS 进程身份不匹配')
whitelist={'PATH','LD_LIBRARY_PATH','LIBRARY_PATH','PYTHONPATH','ASCEND_HOME_PATH',
           'ASCEND_OPP_PATH','ASCEND_TOOLKIT_HOME','CANN_HOME_PATH'}
env={}
for raw in (proc/'environ').read_bytes().split(b'\0'):
    if not raw or b'=' not in raw:continue
    key,value=raw.decode().split('=',1)
    if key in whitelist:env[key]=value
env.update(ASCEND_RT_VISIBLE_DEVICES='7',KGS_FLAGGEMS_ROOT=str(r/'FlagGems-case-adapter'))
args=['docker','exec','-d']
for key,value in env.items():args+=['-e',key+'='+value]
log=r/'kgs-19656.log'
profile_root=r/'kgs-profiles-narrow-master'
# 所有路径必须处于本次独立实验根，且为无 shell 特殊字符的绝对路径。
for path in (r,log,profile_root):
    if not path.is_absolute() or any(ch in str(path) for ch in " '\";$`\\\n"):
        raise SystemExit('实验路径不支持 shell 特殊字符')
args+=['tle_yy','bash','-c',
       f'exec /usr/local/python3.11.15/bin/python -u -m kernelgen_server.server --host 127.0.0.1 --backend npu --timing walltime --max-workers 1 --port {port} --profile-artifact-root {profile_root} >{log} 2>&1']
subprocess.run(args,check=True)
print('STARTED_NARROW_KGS',port,flush=True)
PY
for ((i=0;i<60;i++)); do
 curl --noproxy '*' -fsS "http://127.0.0.1:$port/status" >/dev/null 2>&1 && break
 sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:$port/status" >"$r/narrow-server-status.json"
seed=$p/operators/narrow_copy/candidates/flaggems-master-20261010.py
baseline=$p/operators/narrow_copy/reports/master-case-adapter-20261010
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy --source "$seed" \
 --server "http://127.0.0.1:$port" --output "$baseline" --label flaggems-master --timeout 1800
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --server "http://127.0.0.1:$port" --output "$p/operators/narrow_copy/reports/contiguous-copy-20261010" \
 --label contiguous-copy --timeout 1800 || echo 'CONTIGUOUS_COPY_FAILED'
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/narrow_copy/scripts/probe-current-timing.py" \
 --file "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --server "http://127.0.0.1:$port" \
 --output "$r/diagnostics/narrow-current-timing-20261010" --timeout 900 \
 || echo 'NARROW_TIMING_SCOPE_DIAGNOSTIC_FAILED'
"$py" "$p/tools/evaluation/collect-optimized-profiles.py" narrow_copy \
 "$p/operators/narrow_copy/reports/contiguous-copy-20261010/contiguous-copy-1.result.json" \
 "$p/operators/narrow_copy/reports/optimized-profile-20261010" \
 --server "http://127.0.0.1:$port" --kernel-prefix narrow_copy_contiguous_kernel \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0' \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4' \
 || echo 'NARROW_OPTIMIZED_PROFILE_FAILED'
export KG_RUN_ROOT="$r/kg-runs" KG_SOURCE_ROOT="$r/Ascend-Optimization/sources"
export KG_WALLTIME_PORT=$port KG_PROFILE_PORT=$port
for arm in no-profile profile; do
 bash "$p/tools/evaluation/run-kg-operator-910b.sh" narrow_copy "$arm" "$arm-master-20261010" 2 "$seed" \
  >"$r/campaign-logs/narrow-master-$arm.log" 2>&1 || echo "NARROW_KG_FAILED_$arm"
 "$py" "$p/tools/evaluation/finalize-kg-arm.py" \
  "$r/kg-runs/narrow_copy-$arm-master-20261010" narrow_copy \
  "$p/operators/narrow_copy/reports/kg-$arm-master-20261010" \
  --label "kg-$arm" --server "http://127.0.0.1:$port" \
  || echo "NARROW_KG_INDEPENDENT_FAILED_$arm"
done
# 四版本全部通过后才生成完整对照；任何一组失败都保留原始证据。
"$py" "$p/tools/evaluation/compare-operator-versions.py" \
 --version master "$baseline/flaggems-master-1.result.json" \
 --version kg-no-profile "$p/operators/narrow_copy/reports/kg-no-profile-master-20261010/independent/kg-no-profile-1.result.json" \
 --version kg-profile "$p/operators/narrow_copy/reports/kg-profile-master-20261010/independent/kg-profile-1.result.json" \
 --version optimized "$p/operators/narrow_copy/reports/contiguous-copy-20261010/contiguous-copy-1.result.json" \
 --output "$p/operators/narrow_copy/reports/master-closure-20261010/版本对比.md" \
 || { echo 'NARROW_CLOSURE_INCOMPLETE'; exit 1; }
echo 'NARROW_FOUR_VERSIONS_PASSED'
"$py" "$p/tools/evaluation/summarize-master-goal.py" "$p" \
 "$p/operators/低于0.8算子四版本闭环-20261010.md"
