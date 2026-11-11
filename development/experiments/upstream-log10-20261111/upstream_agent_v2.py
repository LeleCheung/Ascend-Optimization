from pathlib import Path
import json,os,subprocess,hashlib,shutil,sys
root=Path('/data/hanle/ascend-optimization')
repo=root/'Ascend-Optimization'
run=root/'runs/flaggems-upstream-log10-20261111'
work=run/'agent-v2';work.mkdir(exist_ok=False)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=repo/'operators/log10_/versions/independent_strict/retained/kernel.py'
for name in ['candidate.py','log10.py']:shutil.copyfile(source,work/name)
reference=repo/'development/dependencies/third_party/FlagGems'
shutil.copyfile(reference/'src/flag_gems/runtime/backend/_ascend/ops/__init__.py',work/'backend_init.py')
shutil.copyfile(reference/'src/flag_gems/ops/log10.py',work/'upstream_generic_log10.py')
shutil.copytree(run/'method/hgkda/skills/hgkda',work/'skills/hgkda')
save(work/'diagnosis.json',dict(task='upstream_interface_integration',changes=[],unchanged=[],unknown=[]))
prompt='''遵循skills/hgkda/SKILL.md与references/upstream-validation.md。本轮只做固定FlagGems版本d6a8eec473517a3d68157b208eb9c057eb1d4c50的log10_接入，不进行性能调优。
candidate.py是已通过严格Triton审查与KGS/补测的既有交付原件，仅可读。log10.py是其原字节副本，由你修改成FlagGems昇腾后端模块，最终宿主放到src/flag_gems/runtime/backend/_ascend/ops/log10.py。backend_init.py是该后端ops/__init__.py的原件副本，由你更新导出。upstream_generic_log10.py是通用上游模块，供理解接口与许可证格式，仅可读。
只允许修改log10.py、backend_init.py、diagnosis.json。使用Read/Edit工具，不用其他agent，不读工作区外目录，不运行设备测试，不创建外部依赖。不直接调用PyTorch/CANN计算。
导出正确的log10_(A)接口，使flag_gems.log10_在Ascend加载该实现；保持通用log10与log10_out接口不受影响。保持既有Triton kernel、数学、数据布局、分块、编译缓存和当前stream查询逻辑，不借接入进行优化或重写。保留上游许可证声明，使用专业中文说明新增接入。不要将先前既有实现的贡献改写为你本次从零优化。
在diagnosis.json说明改动、保留的机制、必须由宿主测试的项目。宿主会审查源码与分发、执行原生tests和benchmark，不要自行声称通过。完成最小必要接口接入后结束。'''
(work/'prompt.md').write_text(prompt,encoding='utf8')
settings=json.loads((root/'runs/log10-method-transfer-20261110-frozen-v1/model-settings.json').read_bytes())
env=dict(os.environ,CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',CLAUDE_CODE_MAX_OUTPUT_TOKENS='16384',MAX_THINKING_TOKENS='4096')
cmd=[str(root/'runtime/kg-controller/bin/claude'),'-p','--bare','--output-format','stream-json','--verbose',
 '--max-turns','14','--tools','Read,Edit','--allowedTools','Read','Edit','--permission-mode','acceptEdits','--no-session-persistence',
 '--model',settings['model'],'--effort',settings.get('effort','low')]
save(work/'runtime-settings.json',dict(env={k:env[k] for k in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_MAX_OUTPUT_TOKENS','MAX_THINKING_TOKENS']}))
cmd += ['--settings',str(work/'runtime-settings.json')]
save(work/'invocation.json',dict(model=settings['model'],argv=cmd,delivery_sha256=sha(source),prompt_sha256=sha(work/'prompt.md'),
 original_backend_init_sha256=sha(work/'backend_init.py'),scope='model integration only; Codex does not edit candidate'))
with (work/'events.jsonl').open('xb') as out,(work/'stderr.log').open('xb') as err:
 p=subprocess.run(cmd,input=prompt.encode(),stdout=out,stderr=err,cwd=work,env=env,timeout=1200)
records=[]
for line in (work/'events.jsonl').read_bytes().splitlines():
 try:
  item=json.loads(line)
  if item.get('type')=='result':records.append(item)
 except ValueError:pass
result=records[-1] if records else {}
save(work/'model-result.json',dict(returncode=p.returncode,result=result,
 output_sha256={n:sha(work/n) for n in ['log10.py','backend_init.py','diagnosis.json']}))
print(json.dumps(dict(returncode=p.returncode,is_error=result.get('is_error'),subtype=result.get('subtype'),turns=result.get('num_turns'))))
assert p.returncode==0 and result and not result.get('is_error'), 'model integration failed; inspect model-result.json'
