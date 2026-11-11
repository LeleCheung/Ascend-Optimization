from pathlib import Path
import os,json,subprocess,shutil,hashlib
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111';a=w/'agent-recovery';a.mkdir()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(w/'agent-abi-fix/log10.py',a/'log10.py')
for name in ['ordinary-launch-regression.json','profiler-state-probe.txt','profiler-condition-excerpt.txt','profiler-runtime-excerpt.txt']:
 shutil.copyfile(w/name,a/name)
save(a/'diagnosis.json',dict(evidence=[],change='',validation=[]))
prompt='Restore ordinary launch performance while preserving profiler compatibility. Read ordinary-launch-regression.json and profiler-state-probe.txt first, then relevant driver excerpts and log10.py. Current candidate passes upstream correctness 18/18, upstream profiler benchmark 9 points, KGS 36/36, dual reference checks 66+66 and edges 15; but KGS vs CANN is 0.750834 (<0.8). Previous retained direct-launch candidate using integer pointer measured 0.899286; unconditional Tensor argument increased small-input host latency.\nYour responsibility: edit log10.py and diagnosis.json only. Keep exact Triton mathematics, layout branches and stream semantics. Both run and log10_ must bind to the same function. No tests, tolerances, benchmark detection, native compute, CPU fallback or profiler disabling. You may change host metadata/launch representation to satisfy ordinary and profiling API contracts cheaply; kernel math must stay identical.\nEvidence: driver profiler start latches flags; stop does not clear shown flags. Active-only checks are insufficient: an unrelated profile can run without this candidate and next candidate call still requires shape metadata. torch.autograd._profiler_enabled() true inside/false outside; ProfPathCreator().get_prof_dir() stays nonempty after stop on this environment. Use actual supported state information conservatively; do not use result-file existence or test names. Never catch launch error and retry this in-place kernel: launch may already have executed. Consider whether metadata-bearing pointer representation or state/history checking avoids repeated Tensor pointer resolution while preserving correct size metadata. This is a hypothesis, choose based on driver evidence. Unknown states must retain correctness. Retain debug/compile-only guards.\nRead relevant lines efficiently. All implementation by you; host only reviews and runs tests. Write diagnosis with evidence, reasoning, limitations and required validation (ordinary, unrelated profiling, during profiling, after stop across 3 dtypes). No external reads or device execution. Do not seek agents. Use professional comments. Finish after source and diagnosis are complete.'
(a/'prompt.md').write_text(prompt,encoding='utf8')
env=dict(os.environ,CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',CLAUDE_CODE_MAX_OUTPUT_TOKENS='16384',MAX_THINKING_TOKENS='4096')
save(a/'runtime-settings.json',dict(env={k:env[k] for k in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_MAX_OUTPUT_TOKENS','MAX_THINKING_TOKENS']}))
cmd=[str(r/'runtime/kg-controller/bin/claude'),'-p','--bare','--output-format','stream-json','--verbose','--max-turns','16',
 '--tools','Read,Edit','--allowedTools','Read','Edit','--permission-mode','acceptEdits','--no-session-persistence','--model','deepseek-flash',
 '--effort','low','--settings',str(a/'runtime-settings.json')]
save(a/'invocation.json',dict(argv=cmd,input_sha256=sha(a/'log10.py'),prompt_sha256=sha(a/'prompt.md')))
with (a/'events.jsonl').open('xb') as out,(a/'stderr.log').open('xb') as err:
 p=subprocess.run(cmd,input=prompt.encode(),cwd=a,env=env,stdout=out,stderr=err,timeout=900)
result={}
for line in (a/'events.jsonl').read_bytes().splitlines():
 try:
  item=json.loads(line)
  if item.get('type')=='result':result=item
 except ValueError:pass
save(a/'model-result.json',dict(returncode=p.returncode,result=result,output_sha256=sha(a/'log10.py')))
print(json.dumps(dict(returncode=p.returncode,is_error=result.get('is_error'),turns=result.get('num_turns'),sha256=sha(a/'log10.py'))))
