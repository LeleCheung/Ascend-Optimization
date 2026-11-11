from pathlib import Path
import os,json,subprocess,shutil,hashlib
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111';a=w/'agent-profiler-fix';a.mkdir()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(w/'agent-v2/log10.py',a/'log10.py')
shutil.copyfile(w/'upstream-validation-latest.md',a/'method.md')
shutil.copyfile(w/'profiler-runtime-excerpt.txt',a/'runtime-excerpt.txt')
log=(w/'attempt-1/upstream/log10_/performance_stdout.log').read_text()
start=log.find('FAILURES');end=log.find('warnings',start)
(a/'failure.txt').write_text(log[start:end if end>start else start+14000],encoding='utf8')
save(a/'diagnosis.json',dict(evidence=[],hypothesis='',change='',validation=[]))
prompt='''先读method.md、failure.txt、runtime-excerpt.txt，并读取log10.py一次。本轮修复上游profiler benchmark启动兼容，不改变kernel数学、分块或测试。
现象：log10_原生正确性18项通过。原生benchmark在torch_npu.profiler激活后，直接launcher调用报int对象没有size属性。厂商driver中有对Tensor执行size()的代码。请核对快速启动参数是否丢失Tensor元数据；选择能兼容profiling且尽量保留正常启动性能的最小改动。可以考虑保留真实Tensor参数或必要时采用受支持的Triton启动路径，由你决定具体实现。
只修改本目录已有log10.py和diagnosis.json，用Read/Edit。全部计算保持严格Triton。不得关闭profiler、改计时方法、增加原生计算回退、修改kernel数学或放宽测试。必须保留上游log10_(A)和已有run(A)接口、动态地址和当前stream语义。Codex不直接修改候选代码，所有候选修改由你完成。
不要访问工作区外或猜测其他文件，不运行设备测试，不找其他agent。记录证据、改法和需要验证的普通调用/profiler/布局路径；宿主负责审查和执行测试。完成修改后简短返回。'''
(a/'prompt.md').write_text(prompt,encoding='utf8')
env=dict(os.environ,CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',CLAUDE_CODE_MAX_OUTPUT_TOKENS='16384',MAX_THINKING_TOKENS='4096')
save(a/'runtime-settings.json',dict(env={k:env[k] for k in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_MAX_OUTPUT_TOKENS','MAX_THINKING_TOKENS']}))
cmd=[str(r/'runtime/kg-controller/bin/claude'),'-p','--bare','--output-format','stream-json','--verbose','--max-turns','12',
 '--tools','Read,Edit','--allowedTools','Read','Edit','--permission-mode','acceptEdits','--no-session-persistence','--model','deepseek-flash',
 '--effort','low','--settings',str(a/'runtime-settings.json')]
save(a/'invocation.json',dict(argv=cmd,input_sha256=sha(a/'log10.py'),prompt_sha256=sha(a/'prompt.md')))
with (a/'events.jsonl').open('xb') as out,(a/'stderr.log').open('xb') as err:
 p=subprocess.run(cmd,input=prompt.encode(),cwd=a,env=env,stdout=out,stderr=err,timeout=1200)
result={}
for line in (a/'events.jsonl').read_bytes().splitlines():
 try:
  item=json.loads(line)
  if item.get('type')=='result':result=item
 except ValueError:pass
save(a/'model-result.json',dict(returncode=p.returncode,result=result,output_sha256=sha(a/'log10.py')))
print(json.dumps(dict(returncode=p.returncode,is_error=result.get('is_error'),turns=result.get('num_turns'),sha256=sha(a/'log10.py'))))
