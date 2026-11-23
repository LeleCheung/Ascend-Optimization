"""Return complete device results to the candidate model through a read-only call."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gqa-only', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve()
    run = root / 'model-result-review'
    workspace = run / 'workspace'
    workspace.mkdir(parents=True, exist_ok=False)
    inputs = {}
    specs = [] if args.gqa_only else [('group_norm', root / 'native-group-norm-hypothesis', ['unchanged-ir', 'ir-001', 'ir-002']),
        ('value_selecting', root / 'value-selecting-hypothesis', ['unchanged-ir', 'ir-002', 'ir-003'])]
    for name, directory, invocations in specs:
        inputs[name] = dict(selection=read(directory / 'selection.json'), formal_results={})
        for invocation in invocations:
            path = directory / 'evaluations' / invocation / 'result.json'
            inputs[name]['formal_results'][invocation] = dict(path=str(path), sha256=sha(path), result=read(path))
    gqa = root / 'gqa'
    inputs['gqa'] = dict(parent=read(gqa / 'parent-validation/results.json'),
        iterations={path.parent.parent.name: read(path) for path in sorted(
            (gqa / 'iterations').glob('*/parent-validation/results.json'))},
        postcapture={path.parent.name: read(path) for path in sorted(
            (gqa / 'iterations').glob('*/observation-status.json'))},
        retention={path.name: read(path) for path in sorted(gqa.glob('retention-*.json'))},
        selection=read(gqa / 'selection.json') if (gqa / 'selection.json').exists() else read(gqa / 'parent-selection.json'))
    save(workspace / 'complete-results.json', inputs)
    prompt = '''这是同一夜间研究的只读结果回传。使用Read完整读取当前目录complete-results.json，里面保留GroupNorm三版、回填三版以及GQA原版／失败版／修复版的完整设备结果与已保留选择。禁止编辑任何候选、诊断或调度文件，本调用只有Read工具。
用简明中文确认完整结果已收到：列出两项已验收IR目标shape的原／改耗时、全合同耗时几何变化；指出回填ir-003相对已保留ir-002的负结果。核对GQA修复版四项官方checker和真实耗时，计算case4相对原保留版改善；明确case1资源修复恢复原因果分块，三个非因果使用更宽KV分块。原始GQA失败版case4虽变快但case1编译失败；修复版完整通过。GQA后模拟器仍在采集，机制兑现按后续重放决定。说明物理UB估计和部分窗口的证据范围，不外推官方全20项。不得生成或修改目标实现，不引入新测试、ABBA或搜索预算。可提出一条供后续研究的机制建议，单列为建议。'''
    if args.gqa_only:
        prompt = '''这是GQA续接的只读完整结果回传。用Read完整读取complete-results.json，其中包含原四项基线、全部新候选完整结果、后取证状态及保留选择。仅Read，禁止编辑任何文件。
简明确认收到各版完整四项，按真实结果列出case4耗时及官方checker状态；从parent原始结果读取本轮起点，计算保留版相对本轮起点的目标变化，同时说明四项集合表现和负结果。实际源码／IR层与部分重放边界分别说明，完整20项官方验收留作后续。编译资源估计、浮点逐位等价及降低搬运的预测须与实际编译／checker／重放结果区分。可以提出一个后续假设，保持建议身份；本调用只回收完整结果。'''
    (run / 'prompt.md').write_text(prompt, encoding='utf8')
    settings = read(root / 'gqa/model-settings.json')
    environment = dict(os.environ)
    environment.update({key: str(value) for key, value in settings.get('env', {}).items()})
    environment['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '80'
    environment['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(settings.get('max_output_tokens', 16384))
    environment['MAX_THINKING_TOKENS'] = str(settings.get('thinking_tokens', 4096))
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    runtime = {key: environment[key] for key in
        ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS', 'MAX_THINKING_TOKENS']}
    save(run / 'runtime-settings.json', dict(env=runtime))
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
        '--tools', 'Read', '--allowedTools', 'Read', '--permission-mode', 'acceptEdits',
        '--no-session-persistence', '--settings', str(run / 'runtime-settings.json')]
    if settings.get('model'):
        command += ['--model', settings['model']]
    if settings.get('effort'):
        command += ['--effort', settings['effort']]
    save(run / 'invocation.json', dict(command=command, model=settings.get('model'),
        complete_results_sha256=sha(workspace / 'complete-results.json'), readonly_tools=['Read'],
        started_at=datetime.now(timezone.utc).isoformat(), secret_environment_recorded=False))
    with (run / 'agent-events.jsonl').open('xb') as output, (run / 'agent-stderr.log').open('xb') as error:
        process = subprocess.run(command, input=prompt.encode(), cwd=workspace, env=environment,
            stdout=output, stderr=error)
    results = [json.loads(line) for line in (run / 'agent-events.jsonl').read_bytes().splitlines()]
    result = next(row for row in reversed(results) if row.get('type') == 'result')
    save(run / 'model-usage.json', dict(returncode=process.returncode,
        result={key: result.get(key) for key in ['subtype', 'is_error', 'num_turns', 'duration_ms',
            'usage', 'modelUsage', 'total_cost_usd', 'result']}))
    assert process.returncode == 0 and not result.get('is_error')
    (run / 'response.md').write_text(result.get('result', ''), encoding='utf8')
    print(dict(status='complete_results_returned_to_model', readonly=True))


if __name__ == '__main__':
    main()
