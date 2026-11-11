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


def target_met(workspace, digest):
    """只接受绑定当前字节、已完成正式评测及补测的终止反馈。"""
    from hgkda.artifacts import unwrap
    for name in ['next-performance-feedback.json', 'performance-feedback.json']:
        path=workspace/name
        if not path.exists():continue
        feedback=unwrap(read(path))
        return (feedback.get('source_sha256')==digest
                and feedback.get('decision')=='target_met'
                and feedback.get('accepted') is True)
    return False


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
    initial_source=workspace/'candidate.py'
    initial_digest=hashlib.sha256(initial_source.read_bytes()).hexdigest() if initial_source.exists() else None
    if config.get('context_kind')=='operator_transfer':
        from transfer_context import retire_previous
        active=workspace/config.get('active_source','starter.py')
        retire_previous(workspace,hashlib.sha256(active.read_bytes()).hexdigest())
    call = run/'agent-calls'/args.invocation
    call.mkdir(parents=True, exist_ok=False)
    active_input=workspace/config.get('active_source','starter.py')
    (call/'input-source.py').write_bytes(active_input.read_bytes())
    if config.get('method_manifest'):(call/'method-manifest.json').write_bytes(Path(config['method_manifest']).read_bytes())
    call_config=dict(config)
    if config.get('method_manifest'):call_config['method_manifest']=str(call/'method-manifest.json')
    save(call/'config.json',call_config)
    # 模型阶段固定本次方法身份；开发控制器后续切换run/config不会改变活动调用。
    # workspace固定路径同时保持冻结阶段各次调用提示词逐字相同。
    execution_config=workspace/'agent-config.json'
    save(execution_config,call_config)
    config=call_config
    operator=config.get('operator',config['job']['operator'])
    if config.get('context_kind')=='operator_transfer':
        from transfer_context import build
        # 宿主预先刷新，避免模型漏调context后误读上一方法包／源码的残留上下文。
        brief=build(config)
        save(call/'initial-context.json',brief)
        (call/'initial-knowledge.json').write_bytes((workspace/'knowledge.json').read_bytes())
    if config.get('context_kind')!='operator_transfer' and config.get('goal_kind')!='performance_recovery_with_formal_feedback':
        assert all((Path(config['paired_directory'])/side/'collection.json').exists()
                   for side in ['candidate', 'reference'])
    tool_path=Path(config.get('agent_tool_path',repo/'tools/skill_validation/agent_tools.py'))
    tool_command = shlex.join([sys.executable, str(tool_path),
                              '--config', str(execution_config.resolve())])
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
宿主已按当前源码与本次固定方法刷新上下文，先读取context-brief.json、contract.json和knowledge.json，并核对源码身份。工具桥context可重新整理证据；用法已在task.json完整给出，不浏览工具实现、help或workspace外的方法清单，这些由宿主校验。已有候选时直接围绕失败反馈修改，不重复探索环境。按task.json中的算子和合同，从starter.py或当前candidate.py生成并优化candidate.py，公共入口run(A)，其余辅助函数单下划线命名。
所有设备计算、布局处理和回退严格Triton，CANN仅为参考。原始starter可能含非Triton后端，必须由你自行改写，不得保留。Codex不直接修补候选，你负责全部候选实现及编译/语义修复。
依据已落盘workflow与知识，输出diagnosis.json：证据、机制假设、修改、预期、验证与未知项。缺失采集保留未知，不虚构IR或性能。
诊断只保留本轮必要字段与证据摘要，控制在约2000个汉字内；不复制完整历史或整段IR。源码注释说明当前实现即可，不持续追加已被取代的长篇实验叙述。已有文件优先小范围Edit；单次Write保持简短。优先完成候选并调用预检，再补充简短诊断，避免在文档修订上耗尽轮次。读取IR先根据kernel符号、循环、拷贝及缓冲分配定位必要片段，不连续读取整份大文件。
已有正确候选时，本次只实现一个可独立验证的性能机制，不同时重写所有布局与启动路径。优先用Edit落盘小改动，再完善同一机制并交接预检；下一次根据完整性能反馈选择后续机制。
运行时接口的可读摘录路径见context-brief.json的runtime_api.path；优先读取摘录与当前反馈，已有验证且本次未改变的ABI不反复扫描。先落盘最小必要修订，避免把全部轮次消耗在重复读取和诊断说明上。
只编写candidate.py和diagnosis.json，写入或修改使用Write／Edit工具，不使用Bash的heredoc、cat、printf等写文件命令。长实现拆为多次小修改，每次处理一个机制，避免单次重写超过模型输出上限；已有正确布局路径时优先保留它。candidate.py只包含算子实现，不混入输入构造、PyTorch对照、测试或探测脚本。不要预检starter.py，不编写或预检probe脚本；设备环境由工具桥管理，不能自行探测宿主venv或容器。
完成后调用工具桥preflight --source <当前目录candidate.py>，在Bash工具的timeout参数设置600000，不在命令前加timeout包装器。若返回awaiting_source_review，保持当前候选不变并结束本次调用，等待外部只读审查；不要为绕过审查另写源码或反复提交。按失败信息修改新源码，不重复同源码测试。达到task.json的预检要求后返回，供宿主正式KGS评测；这只是模型调用边界，不是优化结束。新写的注释与诊断使用专业中文。
若已有performance-feedback.json或next-performance-feedback.json，优先检查其源码身份与逐case损失，性能未达标继续修改。不得声称预检通过即达到性能目标。正式结果由宿主反馈，不能自行提交或捏造。
若绑定当前源码的正式结果已达任务阈值且全部所需补测通过，保留candidate.py原字节，仅在diagnosis.json记录结论并结束；不为追求更高分无条件再改源码。
工具桥命令：'''+tool_command
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8')+'\n工具桥命令：'+tool_command
    (workspace/'prompt.md').write_text(prompt, encoding='utf8')
    (call/'prompt.md').write_text(prompt, encoding='utf8')
    configuration = read(args.model_settings)
    output_tokens=configuration.get('max_output_tokens',16384)
    thinking_tokens=configuration.get('thinking_tokens',4096)
    environment = dict(os.environ)
    if config.get('package'):
        environment['PYTHONPATH']=str(Path(config['package'])/'src')+os.pathsep+environment.get('PYTHONPATH','')
    environment.update({key: str(value) for key, value in configuration.get('env', {}).items()})
    # 仅修正本次子进程继承的1%压缩阈值，不改宿主全局设置。
    environment['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '80'
    environment['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(output_tokens)
    environment['MAX_THINKING_TOKENS'] = str(thinking_tokens)
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    # 显式settings优先于用户settings.env；避免启动时再次恢复旧的1%值。
    save(call/'runtime-settings.json', dict(env={
         'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '80', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(output_tokens),
         'MAX_THINKING_TOKENS': str(thinking_tokens)}))
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
               '--max-turns', str(args.max_turns), '--tools', 'Read,Write,Edit,Bash',
               '--allowedTools', 'Read', 'Write', 'Edit',
               'Bash('+str(sys.executable)+' '+str(tool_path)+' *)',
               '--permission-mode', 'acceptEdits', '--no-session-persistence',
               '--settings', str(call/'runtime-settings.json')]
    if configuration.get('model'):
        command += ['--model', configuration['model']]
    if configuration.get('effort'):
        command += ['--effort',configuration['effort']]
    save(call/'host-invocation.json', dict(argv=command, model=configuration.get('model'),
         started_at=datetime.now(timezone.utc).isoformat(),
         secret_environment_recorded=False,
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         tool_bridge_sha256=hashlib.sha256(tool_path.read_bytes()).hexdigest(),
         config_sha256=hashlib.sha256((call/'config.json').read_bytes()).hexdigest(),
         prompt_sha256=hashlib.sha256(prompt.encode('utf8')).hexdigest(),
         context_settings=dict(autocompact_percent=80, max_output_tokens=output_tokens,
                               thinking_tokens=thinking_tokens,effort=configuration.get('effort')),
         skill_sha256=hashlib.sha256((workspace/'skills/hgkda/SKILL.md').read_bytes()).hexdigest()))
    save(call/'source-provenance.json',dict(input_source=active_input.name,
        input_sha256=hashlib.sha256(active_input.read_bytes()).hexdigest(),candidate_existed=initial_source.exists()))
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
        final_digest=hashlib.sha256(initial_source.read_bytes()).hexdigest() if initial_source.exists() else None
        model_error=result.get('is_error') or str(result.get('result','')).startswith('API Error:')
        previous=read(workspace/'latest-preflight.json') if (workspace/'latest-preflight.json').exists() else {}
        valid = (process.returncode == 0 and not model_error and final_digest!=initial_digest
                 and previous.get('source_sha256')==final_digest and previous.get('status') == 'passed')
        state.update(status='awaiting_source_review' if valid else 'requires_attention',
                     returncode=process.returncode)
        if process.returncode==0 and not model_error and final_digest==initial_digest and target_met(workspace,final_digest):
            state['status']='generation_target_met'
        if not model_error and previous.get('source_sha256')==final_digest and previous.get('status')=='awaiting_source_review':
            state['status']='awaiting_preflight_source_review'
        if model_error:state['error']='model_result_error: '+str(result.get('result',''))[:500]
    except Exception as error:
        state.update(status='requires_attention', error=type(error).__name__+': '+str(error))
        raise
    finally:
        if initial_source.exists():
            (call/'output-source.py').write_bytes(initial_source.read_bytes())
            provenance=read(call/'source-provenance.json')
            provenance['output_sha256']=hashlib.sha256(initial_source.read_bytes()).hexdigest()
            save(call/'source-provenance.json',provenance)
        save(run/'controller-state.json', state)


if __name__ == '__main__':
    main()
