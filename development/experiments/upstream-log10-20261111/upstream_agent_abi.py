from pathlib import Path
import os,json,subprocess,shutil,hashlib
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111';a=w/'agent-abi-fix';a.mkdir()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(w/'agent-profiler-fix/log10.py',a/'log10.py')
error=json.loads((w/'kgs-fixed/formal-b4668c67379e/evaluation/result.json').read_bytes())['log']
(a/'failure.txt').write_text(error,encoding='utf8')
loader=r/'Ascend-Optimization/development/dependencies/sources/kernelgen_server/kernelgen_server/evaluation/loader.py'
lines=loader.read_text().splitlines();(a/'loader-excerpt.txt').write_text(chr(10).join(lines[285:305]),encoding='utf8')
save(a/'diagnosis.json',dict(evidence=[],change='',validation=[]))
prompt='''这是接口兼容的最小修复。先读failure.txt和loader-excerpt.txt，再看log10.py末尾导出部分（约280行后）。当前源码已通过FlagGems目标文件原生正确性18项及9个profiler性能点，KGS的双参考补测各66项及特殊值15项也通过，但KGS加载器拒收：run与log10_不是同一函数对象。
请仅调整本模块的公共入口绑定，使run(A)与log10_(A)均可调用且符合加载器的同一函数对象要求。避免别名或包装造成递归。保留Tensor参数传给直接launcher的profiler修复，保持所有Triton kernel、宿主计算、缓存、动态地址、当前stream及输出语义；不要进行性能优化，不修改测试或KGS。
只可用Read/Edit修改本目录log10.py、diagnosis.json，不读取外部文件，不运行设备。不要反复读完整文件，不找其他agent。新增注释和诊断用专业中文。修改由你完成，宿主负责审查和完整复验，结束时简短说明。'''
(a/'prompt.md').write_text(prompt,encoding='utf8')
env=dict(os.environ,CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',CLAUDE_CODE_MAX_OUTPUT_TOKENS='16384',MAX_THINKING_TOKENS='4096')
save(a/'runtime-settings.json',dict(env={k:env[k] for k in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_MAX_OUTPUT_TOKENS','MAX_THINKING_TOKENS']}))
cmd=[str(r/'runtime/kg-controller/bin/claude'),'-p','--bare','--output-format','stream-json','--verbose','--max-turns','8',
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
