"""Record CC/DeepSeek TTIR edits while keeping operator source immutable."""
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
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(workspace):
    return {p.relative_to(workspace).as_posix(): sha(p) for p in workspace.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}


def checked(run, invocation):
    workspace = run / 'workspace'
    call = run / 'agent-calls' / invocation
    initial = read(call / 'input-manifest.json')
    actual = files(workspace)
    editable = {row['path'] for row in read(workspace / 'ir-inventory.json')['entries']} | {'diagnosis.json'}
    assert set(actual) == set(initial), 'Workspace file set changed'
    assert all(actual[name] == digest for name, digest in initial.items() if name not in editable), 'Frozen input changed'
    assert actual['frozen.py'] == read(run / 'config.json')['source_sha256']
    return workspace, call, initial, actual, editable


def bridge(run, invocation, action):
    workspace, call, initial, actual, editable = checked(run, invocation)
    diagnosis = read(workspace / 'diagnosis.json')
    if action == 'decision':
        assert all(actual[name] == initial[name] for name in editable if name != 'diagnosis.json'), 'IR edited before decision'
        assert diagnosis['source_sha256'] == actual['frozen.py']
        for name in ['hypothesis', 'change', 'expected', 'validation']:
            assert diagnosis.get(name), 'Diagnosis field required: ' + name
        assert diagnosis.get('selected_layer') == 'single_operator_ir'
        categories = set()
        evidence = []
        for item in diagnosis['evidence']:
            path = (workspace / item['path']).resolve()
            assert path.is_relative_to(workspace.resolve()) and path.is_file()
            categories.add(item['category'])
            evidence.append(dict(item, sha256=sha(path)))
        assert {'paired_profiling', 'simulator', 'source_ir'} <= categories
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), diagnosis=diagnosis,
            evidence=evidence, input_files=initial, decision_tool_sha256=sha(Path(__file__)))
        destination = call / 'before-edit-decision.json'
        assert not destination.exists(), 'Decision already recorded for this invocation'
        save(destination, record)
        print(json.dumps(dict(status='decision_recorded', sha256=sha(destination))))
    else:
        decision = read(call / 'before-edit-decision.json')
        assert decision['input_files'] == initial
        changed = [name for name in editable if name != 'diagnosis.json' and actual[name] != initial[name]]
        assert changed, 'Submit an actual IR change or return an explicit no-change judgment'
        save(call / 'submission.json', dict(status='awaiting_ir_review', source_sha256=actual['frozen.py'],
            changed_ir=changed, output_files=actual, decision_sha256=sha(call / 'before-edit-decision.json')))
        print(json.dumps(dict(status='awaiting_ir_review', changed_ir=changed)))


def generate(run, invocation):
    workspace = run / 'workspace'
    call = run / 'agent-calls' / invocation
    call.mkdir(parents=True, exist_ok=False)
    diagnosis = workspace / 'diagnosis.json'
    (call / 'input-diagnosis.json').write_bytes(diagnosis.read_bytes())
    save(diagnosis, dict(source_sha256=sha(workspace / 'frozen.py'), selected_layer='single_operator_ir',
        evidence=[], hypothesis='', alternatives=[], change='', expected='', validation='', unknown=[]))
    initial = files(workspace)
    save(call / 'input-manifest.json', initial)
    for row in read(workspace / 'ir-inventory.json')['entries']:
        original = workspace / row['path']
        destination = call / 'input-ir' / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
    bridge_prefix = shlex.join([sys.executable, str(Path(__file__).resolve()), '--run', str(run),
        '--invocation', invocation])
    prompt = '''本轮对已完整合格的最佳算子做单算子IR性能续接，保持frozen.py逐字节一致。
你负责实际IR候选与数值／编译修复；Codex负责证据和调度。只在当前工作区读取，禁止调用其他agent。
先读取context.json、profiling.json、hotspots.json、timeline.json、ir-inventory.json、frozen.py，
再根据kernel符号、循环、拷贝与布局检查ir/中的实际TTIR。input文件为当前最佳的真实编译结果；
存在lowered-inventory.json时读取其清单，按符号检查compiled-readonly/内实际ttadapter和bcmlir，
核对疑似冗余是否在后端已消除；这些实际低层产物保持冻结，只编辑清单允许的TTIR。
模拟器窗口的绝对时钟资质、观测范围和未报告泳道按材料解释。热点与等待／空闲均为核心输入。
目标为超过当前最佳；修复旧退步、单纯改编译选项、源码重写各有独立研究身份。
选择当前IR中具体且有执行成本的编译缺陷。判断哪些计算或布局应由编译器优化，并给出可测预测。
保留完整数学语义、特殊值、边界、别名及浮点精度要求；严禁根据测试数据拟合、缓存输出或调用外部计算。
先只修改diagnosis.json，填写source_sha256、selected_layer=single_operator_ir、evidence、hypothesis、
alternatives、change、expected、validation、layer_reason、simulator_incremental_judgment。
evidence每项有path、category、role、finding，category取paired_profiling/simulator/source_ir，
明确引用profiling、热点和流水的具体观察；记录你实际读取的材料。
然后调用下列decision桥，成功后才编辑IR。只用Edit修改diagnosis.json和ir-inventory.json列出的ir/*.ttir；
源文件、清单、context和证据保持冻结。没有Write工具，不调用shell写文件。每次调用选择一个可独立验证的机制，
可编辑同机制需要的多个特化，不能修改特化的函数ABI。原始IR可解析是起点，避免重复全文件读取。
如果后端已消除了疑似冗余，解释观察并保持IR原样交接；如果存在可行改法，用小范围Edit落盘。
完成一项修改后调用submit桥，返回awaiting_ir_review即保持文件不变并结束本次调用。
语义、完整KGS和性能由宿主审查后运行，不能把文档、预检或建议当作收益结果。
性能搜索无总轮数、token、cost或总耗时限制；本次交接是审查与完整评测边界。
新诊断采用专业中文。工具命令：
'''
    prompt += bridge_prefix + ' decision\n' + bridge_prefix + ' submit\n'
    if (workspace / 'feedback.json').exists():
        prompt += '先读取feedback.json中的实际正式与补测结果，保留源码和IR身份，针对反馈选择下一步。\n'
    if (workspace / 'codex-brief.json').exists():
        prompt += '读取codex-brief.json中的宿主证据分析与候选方向，核对实际IR后由你选择具体实现；记录Codex的分析贡献。\n'
    (call / 'prompt.md').write_text(prompt, encoding='utf8')
    settings = read(run / 'model-settings.json')
    environment = dict(os.environ)
    environment.update({key: str(value) for key, value in settings.get('env', {}).items()})
    environment.update(CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(settings.get('max_output_tokens', 16384)),
        MAX_THINKING_TOKENS=str(settings.get('thinking_tokens', 4096)))
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    runtime_settings = dict(env={key: environment[key] for key in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS', 'MAX_THINKING_TOKENS']})
    save(call / 'runtime-settings.json', runtime_settings)
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
        '--tools', 'Read,Edit,Bash', '--allowedTools', 'Read', 'Edit',
        'Bash(' + bridge_prefix + ' *)', '--permission-mode', 'acceptEdits',
        '--no-session-persistence', '--settings', str(call / 'runtime-settings.json')]
    if settings.get('model'):
        command += ['--model', settings['model']]
    if settings.get('effort'):
        command += ['--effort', settings['effort']]
    save(call / 'host-invocation.json', dict(argv=command, model=settings.get('model'),
        started_at=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(Path(__file__)),
        prompt_sha256=sha(call / 'prompt.md'), source_sha256=initial['frozen.py'],
        secret_environment_recorded=False, generation_limits=dict(max_turns=None, deadline=None)))
    with (call / 'agent-events.jsonl').open('xb') as out, (call / 'agent-stderr.log').open('xb') as err:
        process = subprocess.run(command, input=prompt.encode(), stdout=out, stderr=err,
            env=environment, cwd=workspace)
    result = {}
    for line in (call / 'agent-events.jsonl').read_bytes().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'result':
            result = event
    save(call / 'model-usage.json', dict(returncode=process.returncode,
        result={key: result.get(key) for key in ['subtype', 'is_error', 'num_turns', 'duration_ms',
            'usage', 'modelUsage', 'total_cost_usd', 'result']}))
    _, _, _, actual, _ = checked(run, invocation)
    for row in read(workspace / 'ir-inventory.json')['entries']:
        path = workspace / row['path']
        destination = call / 'output-ir' / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (call / 'output-diagnosis.json').write_bytes(diagnosis.read_bytes())
    error = process.returncode != 0 or result.get('is_error')
    status = 'requires_attention' if error else ('awaiting_ir_review' if (call / 'submission.json').exists() else 'model_no_submission')
    save(run / 'status.json', dict(status=status, invocation=invocation, source_sha256=actual['frozen.py']))
    print(json.dumps(dict(status=status, invocation=invocation)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('action', choices=['generate', 'decision', 'submit'])
    args = parser.parse_args()
    if args.action == 'generate':
        generate(args.run, args.invocation)
    else:
        bridge(args.run, args.invocation, args.action)


if __name__ == '__main__':
    main()
