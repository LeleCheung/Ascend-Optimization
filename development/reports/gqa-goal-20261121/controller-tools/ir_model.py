"""Record CC/DeepSeek TTIR edits while keeping operator source immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
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
        wording = json.dumps(diagnosis, ensure_ascii=False)
        assert all(phrase not in wording for phrase in ['不' + '是', '不' + '等于', '不能' + '证明']), \
            'Use affirmative concrete statements for evidence, scope and next actions; revise diagnosis wording'
        for name in ['hypothesis', 'change', 'expected', 'validation']:
            assert diagnosis.get(name), 'Diagnosis field required: ' + name
        assert diagnosis.get('selected_layer') == 'single_operator_ir'
        categories = set()
        evidence = []
        for item in diagnosis['evidence']:
            path = (workspace / item['path']).resolve()
            assert path.is_relative_to(workspace.resolve()) and path.is_file(), \
                'Evidence path must name a real workspace file; put sections and field names in finding'
            categories.add(item['category'])
            evidence.append(dict(item, sha256=sha(path)))
        context = read(workspace / 'context.json')
        required = ({'evaluation', 'source_ir'} if context.get('evidence_mode') == 'compile-repair'
                    else {'paired_profiling', 'simulator', 'source_ir'})
        assert required <= categories
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), diagnosis=diagnosis,
            diagnosis_sha256=sha(workspace / 'diagnosis.json'),
            evidence=evidence, input_files=initial, decision_tool_sha256=sha(Path(__file__)))
        destination = call / 'before-edit-decision.json'
        assert not destination.exists(), 'Decision already recorded for this invocation'
        save(destination, record)
        print(json.dumps(dict(status='decision_recorded', sha256=sha(destination))))
    else:
        decision = read(call / 'before-edit-decision.json')
        assert decision['input_files'] == initial
        assert diagnosis == decision['diagnosis'], 'Diagnosis changed after decision; start a new invocation from the retained input'
        if 'diagnosis_sha256' in decision:
            assert sha(workspace / 'diagnosis.json') == decision['diagnosis_sha256'], 'Registered diagnosis bytes changed'
        changed = [name for name in editable if name != 'diagnosis.json' and actual[name] != initial[name]]
        if action == 'no-change':
            assert not changed, 'No-change handoff requires the original IR bytes'
            save(call / 'no-change.json', dict(status='model_no_change', source_sha256=actual['frozen.py'],
                decision_sha256=sha(call / 'before-edit-decision.json'), output_files=actual))
            print(json.dumps(dict(status='model_no_change')))
            return
        assert changed, 'Use the no-change bridge for an explicit unchanged-IR judgment'
        save(call / 'submission.json', dict(status='awaiting_ir_review', source_sha256=actual['frozen.py'],
            changed_ir=changed, output_files=actual, decision_sha256=sha(call / 'before-edit-decision.json')))
        print(json.dumps(dict(status='awaiting_ir_review', changed_ir=changed)))


def generate(run, invocation):
    workspace = run / 'workspace'
    call = run / 'agent-calls' / invocation
    call.mkdir(parents=True, exist_ok=False)
    context = read(workspace / 'context.json')
    editable_stage = context.get('editable_ir_stage', 'ttir')
    assert editable_stage in {'ttir', 'bcmlir', 'hivm'}
    diagnosis = workspace / 'diagnosis.json'
    (call / 'input-diagnosis.json').write_bytes(diagnosis.read_bytes())
    save(diagnosis, dict(source_sha256=sha(workspace / 'frozen.py'), selected_layer='single_operator_ir',
        evidence=[], hypothesis='', alternatives=[], change='', expected='', validation='', unknown=[]))
    initial = files(workspace)
    save(call / 'input-manifest.json', initial)
    # Preserve exact evidence and host feedback bytes before the model reads them.
    shutil.copytree(workspace, call / 'input-workspace',
                    ignore=shutil.ignore_patterns('__pycache__'))
    assert files(call / 'input-workspace') == initial
    for row in read(workspace / 'ir-inventory.json')['entries']:
        original = workspace / row['path']
        destination = call / 'input-ir' / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
    bridge_prefix = shlex.join([sys.executable, str(Path(__file__).resolve()), '--run', str(run),
        '--invocation', invocation])
    prompt = '''本轮对context.json标明的输入版本做单算子IR性能研究，保持frozen.py逐字节一致。
默认从完整合格最佳继续；若context明确标注未保留探索版的修复分支，按该父版本修复，合格最佳仍为验收比较起点。
你负责实际IR候选与数值／编译修复；Codex负责证据和调度。只在当前工作区读取，禁止调用其他agent。
先读取context.json、profiling.json、hotspots.json、timeline.json、ir-inventory.json、frozen.py，
再根据kernel符号、循环、拷贝与布局检查ir/中的实际TTIR。input文件为当前最佳的真实编译结果；
存在lowered-inventory.json时读取其清单，按符号检查compiled-readonly/内实际ttadapter和bcmlir，
核对疑似冗余是否在后端已消除；这些实际低层产物保持冻结，只编辑清单允许的TTIR。
模拟器窗口的绝对时钟资质、观测范围和未报告泳道按材料解释。热点与等待／空闲均为核心输入。
流水累计时间可重叠，热点排名用于定位候选机制；关键路径、设备波次与带宽结论须引用独立支持材料。
前后窗口分别记录覆盖量与归一化范围，动态总次数的比较须核对执行覆盖；未报告泳道保持未知。
每次性能结果约束对应的实际候选与测量范围；把更广的机制排除保留为待验证假设。
目标为超过当前最佳；修复旧退步、单纯改编译选项、源码重写各有独立研究身份。
选择当前IR中具体且有执行成本的编译缺陷。判断哪些计算或布局应由编译器优化，并给出可测预测。
保留完整数学语义、特殊值、边界、别名及浮点精度要求；严禁根据测试数据拟合、缓存输出或调用外部计算。
先只修改diagnosis.json，填写source_sha256、selected_layer=single_operator_ir、evidence、hypothesis、
alternatives、change、expected、validation、layer_reason、simulator_incremental_judgment。
layer_reason分别解释Triton表达能力、固定源码实验目的与IR控制点；同类意图可在源码表达时写清替代方向。
expected逐项列出预测的指令、动态轮次、等待与资源变化，宿主后重放将分别核对兑现情况及偏差。
evidence每项有path、category、role、finding，category取paired_profiling/simulator/source_ir，
明确引用profiling、热点和流水的具体观察；记录你实际读取的材料。
然后调用下列decision桥，成功后才编辑IR。只用Edit修改diagnosis.json和ir-inventory.json列出的ir/*.ttir；
decision成功后diagnosis.json也保持原字节，实际编辑须与已登记机制一致。新增观察写入最终回复。
若选择保持IR原样，登记判断后调用no-change桥并结束；该交接有效，宿主继续搜索。
若登记后改变机制，保持本轮原件，交回宿主后另建调用重新登记，再由模型修改IR。
源文件、清单、context和证据保持冻结。没有Write工具，不调用shell写文件。每次调用选择一个可独立验证的机制，
可编辑同机制需要的多个特化，不能修改特化的函数ABI。原始IR可解析是起点，避免重复全文件读取。
如果后端已消除了疑似冗余，解释观察并保持IR原样交接；如果存在可行改法，用小范围Edit落盘。
完成一项修改后调用submit桥，返回awaiting_ir_review即保持文件不变并结束本次调用。
语义、完整KGS和性能由宿主审查后运行，不能把文档、预检或建议当作收益结果。
性能搜索无总轮数、token、cost或总耗时限制；本次交接是审查与完整评测边界。
新诊断采用专业中文。工具命令：
'''
    prompt += '中文采用正向具体陈述；禁用短语按不+是、不+等于、不能+证明拼接，decision桥会校验措辞。\n'
    prompt += bridge_prefix + ' decision\n' + bridge_prefix + ' submit\n' + bridge_prefix + ' no-change\n'
    if editable_stage == 'bcmlir':
        prompt = prompt.replace('实际TTIR', '实际BCMLIR')
        prompt = prompt.replace('只编辑清单允许的TTIR', '只编辑清单允许的BCMLIR')
        prompt = prompt.replace('ir/*.ttir', 'ir/*.bcmlir')
    if editable_stage == 'hivm':
        prompt = prompt.replace('实际TTIR', '实际HIVM')
        prompt = prompt.replace('只编辑清单允许的TTIR', '只编辑清单允许的HIVM')
        prompt = prompt.replace('ir/*.ttir', 'ir/*.mlir')
        prompt += ('\n本次编辑编译器实际导出的HIVM，原输入已通过独立后端续编译与实测二进制关联。'
                   '严格审查显式buffer地址、别名、跨迭代生存期、事件配对和跨核同步；'
                   '保持函数ABI及未涉及的核心原样。所有修改由宿主逐项审查，再编译和真机评测。\n')
    if context.get('evidence_mode') == 'compile-repair':
        prompt = prompt.replace('input文件为当前最佳的真实编译结果',
            'input文件为当前探索分支真实编译失败的产物；保留最佳的历史材料单独标识')
        prompt = prompt.replace('category取paired_profiling/simulator/source_ir',
            'category取evaluation/source_ir/paired_profiling/simulator')
        prompt += ('\n当前为明确未保留源码分支的编译修复研究。先读取feedback.json和codex-brief.json，'
            '当前目标IR已实际编译失败，profiling/hotspots/timeline均标为历史版本材料。'
            '修改前证据必须包含evaluation与source_ir，按实际来源记录历史材料；'
            '当前源码保持逐字节冻结，编译修复、同源码IR性能收益与超过保留最佳分别验收。'
            '本调用只提交IR，CPU编译与完整原四项真机检查随后由宿主执行。\n')
    if (workspace / 'feedback.json').exists():
        prompt += '先读取feedback.json中的实际正式与补测结果，保留源码和IR身份，针对反馈选择下一步。\n'
    if (workspace / 'codex-brief.json').exists():
        prompt += '读取codex-brief.json中的宿主证据分析与候选方向，核对实际IR后由你选择具体实现；记录Codex的分析贡献。\n'
        requested = read(workspace / 'codex-brief.json').get('requested_mechanism')
        if requested:
            prompt += ('本调用限定检验下述机制，具体实现由你决定；若证据或语义审查阻止实施，'
                       '登记原因后按no-change交接，其他机制留给独立调用：\n' + requested + '\n')
    if (workspace / 'validation-policy.json').exists():
        prompt += '读取validation-policy.json；复用固定完整合同及既有补测，针对已登记具体风险复用现有诊断。ABBA按测量疑点选用，SHA由工具自动记录，候选作者专注机制和实现。\n'
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
        '--tools', 'Read,Grep,Glob,Edit,Bash', '--allowedTools', 'Read', 'Grep', 'Glob', 'Edit',
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
    save(run / 'status.json', dict(status='model_generation_running', invocation=invocation,
         source_sha256=initial['frozen.py'], host_invocation=str(call / 'host-invocation.json')))
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
    status = 'requires_attention' if error else ('awaiting_ir_review' if (call / 'submission.json').exists()
        else 'model_no_change' if (call / 'no-change.json').exists() else 'model_no_submission')
    save(run / 'status.json', dict(status=status, invocation=invocation, source_sha256=actual['frozen.py']))
    print(json.dumps(dict(status=status, invocation=invocation)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('action', choices=['generate', 'decision', 'submit', 'no-change'])
    args = parser.parse_args()
    if args.action == 'generate':
        handoff = args.run / ('generation-handoff-' + args.invocation + '.json')
        if handoff.exists():
            record = read(handoff)
            target = Path(record['run']).resolve()
            assert target.is_relative_to(args.run.resolve().parent)
            assert read(target / 'config.json')['source_sha256'] == read(args.run / 'config.json')['source_sha256']
            assert record['invocation'] == args.invocation
            print(json.dumps(dict(status='generation_handed_off', **record)))
            return
        generate(args.run, args.invocation)
    else:
        bridge(args.run, args.invocation, args.action)


if __name__ == '__main__':
    main()
