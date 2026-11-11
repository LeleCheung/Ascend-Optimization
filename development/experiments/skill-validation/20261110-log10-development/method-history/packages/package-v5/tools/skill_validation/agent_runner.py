"""让宿主模型加载运行目录内的独立 skill；不重采集、不执行正式评测。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-settings', type=Path, required=True)
    parser.add_argument('--invocation', default='first')
    parser.add_argument('--max-turns', type=int, default=36)
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    run = Path(config['run_directory'])
    repo = Path(config['repository'])
    workspace = run/'workspace'
    call = run/'agent-calls'/args.invocation
    call.mkdir(parents=True, exist_ok=False)
    operator=config.get('operator',config['job']['operator'])
    if config.get('context_kind')!='operator_transfer' and config.get('goal_kind')!='performance_recovery_with_formal_feedback':
        assert all((Path(config['paired_directory'])/side/'collection.json').exists()
                   for side in ['candidate', 'reference'])
    tool_path=Path(config.get('agent_tool_path',repo/'tools/skill_validation/agent_tools.py'))
    tool_command = shlex.join([sys.executable, str(tool_path),
                              '--config', str(args.config.resolve())])
    save(workspace/'task.json', dict(operator=operator, source=config.get('active_source','starter.py'),
         baseline_source='starter.py',
         required_formal_cases=config.get('formal_cases',36),
         required_supplementary_each=config['job']['checks']['supplementary']['previous_total'], tool_command=tool_command,
         tool_actions=dict(context=tool_command+' context',
                           preflight=tool_command+' preflight --source '+str(workspace/'candidate.py')),
         environment='宿主仅运行工具桥；设备依赖由桥接器在容器内调用，不在宿主venv探测Triton',
         initial_evidence_policy='原始起点含目标设备非Triton回退时，先由模型建立合规候选，再采集其IR及成对profiling；不以缺失采集阻塞初始适配',
         feedback_file=('next-performance-feedback.json' if (workspace/'next-performance-feedback.json').exists()
                        else 'performance-feedback.json'),
         optimization_stop='完整正式性能超过阈值且所需补测通过；模型调用返回不代表优化任务完成',
         public_entrypoint='run(A)；其余Python调用函数以单下划线命名，避免KGS公共符号冲突',
         candidate_file='candidate.py',device_probe='工具桥仅预检候选实现，不执行任意测试脚本；不要提交starter.py或probe脚本',
         current_best_source_read_allowed=False,phase=config.get('phase','development'),
         method_manifest_sha256=(hashlib.sha256(Path(config['method_manifest']).read_bytes()).hexdigest() if config.get('method_manifest') else None),
         target='同CANN参考完整调用争取超过0.8，至少相对固定起点产生有效优化'))
    prompt = '''加载并遵循 skills/hgkda/SKILL.md，任务见 task.json。只在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他agent。
先调用工具桥context，读取context-brief.json、contract.json和knowledge.json。工具桥用法已在task.json完整给出，不浏览工具实现、help或workspace外的方法清单；这些由宿主校验。已有候选时直接围绕失败反馈修改，不重复探索环境。按task.json中的算子和合同，从starter.py或当前candidate.py生成并优化candidate.py，公共入口run(A)，其余辅助函数单下划线命名。
所有设备计算、布局处理和回退严格Triton，CANN仅为参考。原始starter可能含非Triton后端，必须由你自行改写，不得保留。Codex不直接修补候选，你负责全部候选实现及编译/语义修复。
依据已落盘workflow与知识，输出diagnosis.json：证据、机制假设、修改、预期、验证与未知项。缺失采集保留未知，不虚构IR或性能。
只编写candidate.py和diagnosis.json，写入或修改使用Write／Edit工具，不使用Bash的heredoc、cat、printf等写文件命令。candidate.py只包含算子实现，不混入输入构造、PyTorch对照、测试或探测脚本。不要预检starter.py，不编写或预检probe脚本；设备环境由工具桥管理，不能自行探测宿主venv或容器。
完成后调用工具桥preflight --source <当前目录candidate.py>，在Bash工具的timeout参数设置600000，不在命令前加timeout包装器。若返回awaiting_source_review，保持当前候选不变并结束本次调用，等待外部只读审查；不要为绕过审查另写源码或反复提交。按失败信息修改新源码，不重复同源码测试。达到task.json的预检要求后返回，供宿主正式KGS评测；这只是模型调用边界，不是优化结束。新写的注释与诊断使用专业中文。
若已有performance-feedback.json或next-performance-feedback.json，优先检查其源码身份与逐case损失，性能未达标继续修改。不得声称预检通过即达到性能目标。正式结果由宿主反馈，不能自行提交或捏造。
工具桥命令：'''+tool_command
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8')+'\n工具桥命令：'+tool_command
    (workspace/'prompt.md').write_text(prompt, encoding='utf8')
    (call/'prompt.md').write_text(prompt, encoding='utf8')
    configuration = read(args.model_settings)
    environment = dict(os.environ)
    if config.get('package'):
        environment['PYTHONPATH']=str(Path(config['package'])/'src')+os.pathsep+environment.get('PYTHONPATH','')
    environment.update({key: str(value) for key, value in configuration.get('env', {}).items()})
    # 仅修正本次子进程继承的1%压缩阈值，不改宿主全局设置。
    environment['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '80'
    environment['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = '8192'
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    # 显式settings优先于用户settings.env；避免启动时再次恢复旧的1%值。
    save(call/'runtime-settings.json', dict(env={
         'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '80', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '8192'}))
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
               '--max-turns', str(args.max_turns), '--tools', 'Read,Write,Edit,Bash',
               '--allowedTools', 'Read', 'Write', 'Edit',
               'Bash('+str(sys.executable)+' '+str(tool_path)+' *)',
               '--permission-mode', 'acceptEdits', '--no-session-persistence',
               '--settings', str(call/'runtime-settings.json')]
    if configuration.get('model'):
        command += ['--model', configuration['model']]
    save(call/'host-invocation.json', dict(argv=command, model=configuration.get('model'),
         secret_environment_recorded=False,
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         tool_bridge_sha256=hashlib.sha256(tool_path.read_bytes()).hexdigest(),
         config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
         prompt_sha256=hashlib.sha256(prompt.encode('utf8')).hexdigest(),
         context_settings=dict(autocompact_percent=80, max_output_tokens=8192),
         skill_sha256=hashlib.sha256((workspace/'skills/hgkda/SKILL.md').read_bytes()).hexdigest()))
    state = read(run/'controller-state.json')
    state.update(status='agent_generating_and_preflighting',
                 resumed_at=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    state.pop('returncode', None)
    save(run/'controller-state.json', state)
    try:
        with (call/'agent-events.jsonl').open('xb') as out, (call/'agent-stderr.log').open('xb') as err:
            process = subprocess.run(command, input=prompt.encode(), stdout=out, stderr=err,
                                     env=environment, cwd=workspace, timeout=2400)
        result = {}
        for line in (call/'agent-events.jsonl').read_bytes().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('type') == 'result':
                result = record
        save(call/'model-usage.json', dict(returncode=process.returncode,
             result={key: result.get(key) for key in ['subtype', 'is_error', 'num_turns', 'duration_ms',
                     'usage', 'modelUsage', 'total_cost_usd', 'result']}))
        valid = (process.returncode == 0 and (workspace/'latest-preflight.json').exists()
                 and read(workspace/'latest-preflight.json')['status'] == 'passed')
        state.update(status='awaiting_source_review' if valid else 'requires_attention',
                     returncode=process.returncode)
        if (workspace/'latest-preflight.json').exists() and read(workspace/'latest-preflight.json')['status']=='awaiting_source_review':
            state['status']='awaiting_preflight_source_review'
    except Exception as error:
        state.update(status='requires_attention', error=type(error).__name__+': '+str(error))
        raise
    finally:
        save(run/'controller-state.json', state)


if __name__ == '__main__':
    main()
