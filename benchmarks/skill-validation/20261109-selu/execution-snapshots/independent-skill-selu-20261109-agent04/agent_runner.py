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
    assert all((Path(config['paired_directory'])/side/'collection.json').exists()
               for side in ['candidate', 'reference'])
    tool_command = shlex.join([sys.executable, str(repo/'tools/skill_validation/agent_tools.py'),
                              '--config', str(args.config.resolve())])
    save(workspace/'task.json', dict(operator='selu_', source='starter.py',
         required_formal_cases=36, required_supplementary_each=42, tool_command=tool_command,
         current_best_source_read_allowed=False,
         target='同CANN参考完整调用争取超过0.8，至少相对固定起点产生有效优化'))
    prompt = '''加载并遵循 skills/hgkda/SKILL.md，任务见 task.json。仅在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他 agent。
先调用工具桥的 context 动作读取同输入 CANN 成对 profiling 及实际 IR 摘要；优先读取 context-brief.json 和 actual-ir 内的文件，每次读取不超过100行，避免读取完整大 JSON。
从 starter.py 优化 selu_，输出 candidate.py 及 diagnosis.json。所有设备计算和搬运必须 Triton，禁止原生/CANN/ATen fallback，包括非连续输入。
诊断需包含证据、机制、具体修改、预期变化和未决问题。可调整连续分块、精度、非连续寻址和启动组织。
调用工具桥 preflight --source <当前目录candidate.py>，Bash timeout 设置600000。根据失败信息修复，最多三份不同候选，不重复同源码测试。全部42项预检通过后停止。
先不要提交正式评测或虚构成绩；宿主逐路径审查后将调用 KGS，并反馈正式结果。最终简短说明候选文件及预检状态。
工具桥命令：'''+tool_command
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8')+'\n工具桥命令：'+tool_command
    (workspace/'prompt.md').write_text(prompt, encoding='utf8')
    (call/'prompt.md').write_text(prompt, encoding='utf8')
    configuration = read(args.model_settings)
    environment = dict(os.environ)
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
               'Bash('+str(sys.executable)+' '+str(repo/'tools/skill_validation/agent_tools.py')+' *)',
               '--permission-mode', 'acceptEdits', '--no-session-persistence',
               '--settings', str(call/'runtime-settings.json')]
    if configuration.get('model'):
        command += ['--model', configuration['model']]
    save(call/'host-invocation.json', dict(argv=command, model=configuration.get('model'),
         secret_environment_recorded=False,
         context_settings=dict(autocompact_percent=80, max_output_tokens=8192),
         skill_sha256=hashlib.sha256((workspace/'skills/hgkda/SKILL.md').read_bytes()).hexdigest()))
    state = read(run/'controller-state.json')
    state.update(status='agent_generating_and_preflighting',
                 resumed_at=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
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
    except Exception as error:
        state.update(status='requires_attention', error=type(error).__name__+': '+str(error))
        raise
    finally:
        save(run/'controller-state.json', state)


if __name__ == '__main__':
    main()
