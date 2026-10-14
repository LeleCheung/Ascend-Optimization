#!/usr/bin/env python3
"""持续执行采集、CC 建议与候选生成、完整评测和最佳版本复验。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
EVAL_TOOLS = PROJECT / 'tools/evaluation'
sys.path.insert(0, str(EVAL_TOOLS))
from evaluate_operator_import import call
import ascend_profiling_workflow as workflow

spec = importlib.util.spec_from_file_location('comparison', EVAL_TOOLS / 'compare-operator-versions.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def knowledge_for(operator, common):
    text = common.read_text(encoding='utf-8')
    specific = PROJECT / 'operators' / operator / '优化知识.md'
    if specific.is_file():
        text += '\n\n' + specific.read_text(encoding='utf-8')
    return text


def run(command, log, *, prompt=None, timeout=1800, cwd=None, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('wb') as stream:
        process = subprocess.Popen(command, stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                   start_new_session=os.name == 'posix')
        try:
            process.communicate(input=prompt.encode() if prompt is not None else None, timeout=timeout)
        except BaseException:
            # docker exec 的子进程不会随 tmux 自动退出；仅清理此调用创建的进程组。
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f'命令失败 {process.returncode}，见 {log}')


class InterruptedDeviceRequest(RuntimeError):
    """请求已经写入但结果未知，需检查服务而不能自动再次提交。"""


def assert_card_idle():
    for port in (19655, 19657):
        scheduler = call('http://127.0.0.1:' + str(port), '/status', timeout=30)['scheduler']
        if (scheduler['broken'] or scheduler['active'] or scheduler['waiting']
                or scheduler['healthy'] != 1):
            raise InterruptedDeviceRequest(f'物理卡7服务 {port} 尚忙或异常')


def check_pending_evaluation(output, label, repeats):
    for index in range(1, repeats + 1):
        result = output / f'{label}-{index}.result.json'
        request = output / f'{label}-{index}.request.json'
        if request.exists() and not result.exists():
            raise InterruptedDeviceRequest(f'评测请求结果未知，停止自动重发：{request}')


def evaluate(operator, source, output, label, server, repeats=1, *, offline=False):
    check_pending_evaluation(output, label, repeats)
    existing = [output / f'{label}-{index}.result.json' for index in range(1, repeats + 1)]
    if all(path.exists() for path in existing):
        expected_digest = digest(source.read_text(encoding='utf-8'))
        contract = None
        for path in existing:
            result, _ = checked(path, contract)
            if result['source_sha256'] != expected_digest:
                raise ValueError('已有评测对应不同源码，请使用新批次')
            contract = result
        return existing[0]
    if offline:
        raise ValueError('离线报告缺少完整评测结果，不提交设备请求：' + str(output))
    assert_card_idle()
    run([sys.executable, str(EVAL_TOOLS / 'evaluate-operator.py'),
         '--operator', operator, '--source', str(source), '--output', str(output),
         '--label', label, '--server', server, '--repeats', str(repeats),
         '--timeout', '1800', '--timing-scope', 'device_kernel'],
        output.parent / (output.name + '.log'), timeout=4000 * repeats)
    return output / (label + '-1.result.json')


def checked(path, expected=None):
    result = comparison.load_result(path)
    if expected is not None:
        comparison.compare([('contract', expected), ('candidate', result)])
    value = statistics.geometric_mean(r['reference_latency_ms'] / r['latency_ms']
                                      for r in result['timing'].values())
    return result, value


def prefixes(source):
    tree = ast.parse(source)
    result = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any('jit' in ast.unparse(x) for x in node.decorator_list):
            result.append(node.name)
    return result


def profile_best(operator, evaluation, directory, server, count):
    loaded, _ = checked(evaluation)
    rows = sorted(loaded['timing'].values(), key=lambda r: r['speedup'])
    # 相同 dtype/shape 的重复 benchmark case 不占用新的采集名额。
    selected, seen = [], set()
    for row in rows:
        key = json.dumps(row.get('axes'), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            break
    label = evaluation.name.removesuffix('.result.json').rsplit('-', 1)[0]
    request_path = evaluation.with_name(evaluation.name.replace('.result.json', '.request.json'))
    source_path = evaluation.with_name(label + '.py')
    req = json.loads(request_path.read_text(encoding='utf-8'))
    inspection = evaluation.parent / 'inspect.json'
    profile_arguments = []
    directory.mkdir(exist_ok=True)
    failures = []
    for index, row in enumerate(selected):
        output = directory / ('case-' + str(index))
        response = output / 'response.json'
        if not response.exists():
            if output.exists():
                raise InterruptedDeviceRequest(f'先前采集中断，停止自动重发：{output}')
            try:
                assert_card_idle()
                run([sys.executable, str(EVAL_TOOLS / 'profile-operator.py'),
                     '--evaluation', str(evaluation), '--request', str(request_path),
                     '--inspect', str(inspection), '--case-id', row['uuid'], '--output', str(output),
                     '--server', server, '--aic-metrics', 'PipeUtilization', '--timeout', '900'],
                    directory / ('case-' + str(index) + '.log'), timeout=1200)
            except InterruptedDeviceRequest:
                raise
            except Exception as error:
                failures.append({'case': row['uuid'], 'reason': str(error)})
                continue
        data = json.loads(response.read_text(encoding='utf-8'))
        if data.get('status') == 'completed':
            profile_arguments.append((response, output / 'request.json', output / 'artifacts'))
    report = workflow.analyze(operator, source_path, evaluation, request_path, inspection,
                              profile_arguments, timing_scope='device_kernel',
                              evaluation_provenance=evaluation.with_name(label + '.provenance.json'))
    # 源码符号会在候选间变化；只从实际源码 jit 函数选择本次响应的 kernel。
    allowed = prefixes(source_path.read_text(encoding='utf-8'))
    for case in report['cases']:
        metric = case['profiles'].get('metrics')
        if metric is None:
            continue
        response = json.loads(Path(metric['response']).read_text(encoding='utf-8'))
        matches = [op for op in response.get('metrics', {}).get('ops', [])
                   if any(op.get('op_name', '').startswith(p + '_') or op.get('op_name') == p for p in allowed)]
        if matches:
            op = max(matches, key=lambda r: r.get('avg_duration_us') or 0)
            metric['kernel'] = op
            case['metrics'] = {key: op.get(key) for key in (
                'aiv_mte2_ratio', 'aiv_mte3_ratio', 'aiv_scalar_ratio', 'aiv_vec_ratio',
                'aic_mte2_ratio', 'aic_mte1_ratio', 'aic_scalar_ratio', 'aic_mac_ratio', 'cube_utilization_pct')}
            case['metrics'].update(kernel=op.get('op_name'), device_duration_us=op.get('avg_duration_us'),
                                   scope='独立真机 msprof 采集')
            case['diagnosis'] = workflow.diagnose(case)
    report['collection_failures'] = failures
    save(directory / 'diagnosis.json', report)
    (directory / '诊断.md').write_text(workflow.render(report), encoding='utf-8')
    return report


def parse_candidate(response):
    # CC 会在 JSON 前输出 unknown-model 信息；只接受末尾完整 result 对象。
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        values = []
        for line in response.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get('type') == 'result':
                values.append(value)
        if len(values) != 1:
            raise ValueError('无法识别唯一 CC result JSON')
        value = values[0]
    if value.get('is_error'):
        raise ValueError('CC 返回错误')
    structured = value.get('structured_output')
    if structured is None:
        text = value.get('result', '')
        if text.startswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        structured = json.loads(text)
    source = structured['source']
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    aliases = {target.id: node.value.id for node in tree.body if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Name) for target in node.targets if isinstance(target, ast.Name)}
    has_run = 'run' in functions or aliases.get('run') in functions
    if not has_run:
        # DeepSeek sometimes returns the operator-named public function while
        # omitting the repository's required alias. Preserve its complete
        # implementation and add only the adapter entrypoint.
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if 'logaddexp2' in names:
            source = source.rstrip() + '\n\nrun = logaddexp2\n'
            structured['source'] = source
            tree = ast.parse(source)
        else:
            raise ValueError('缺少 run 入口')
    # 只拒绝明确的投机实现；完整正确性及 KGS admission 是最终门禁。
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(x.name.split('.')[0] in {'os', 'subprocess', 'socket', 'requests'} for x in node.names):
                raise ValueError('候选包含系统或网络模块')
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in {'os', 'subprocess', 'socket'}:
            raise ValueError('候选包含系统或网络模块')
    return structured


def prompt_for(operator, source, report, history, knowledge):
    facts = [{'case_id': c['case_id'], 'axes': c['axes'], 'timing': c['walltime'],
              'workload': c['workload_model'], 'metrics': c['metrics'], 'diagnosis': c['diagnosis']}
             for c in report['cases']]
    return '\n'.join([
        '你是 Ascend 910B Triton 算子优化工程师。根据本轮真实诊断产出一个更快的完整单文件候选。',
        '禁止调用 Bash、文件系统、网络或任何工具；不要自行调查环境。下面已经给出完整源码、事实和知识，只在这些内容上推理并直接输出结果。',
        '输出 JSON 对象，仅含 source（完整 Python 源码）、hypothesis（中文修改理由）、expected_effect（中文预期）。',
        '必须保留 run API、所有 dtype、shape、stride、正确性和完整合同。只写目标 Triton 实现，不调用原算子的 PyTorch/FlagGems 作为替代，不删用例，不读取评测环境，不更改计时。',
        '已有候选中的稳定 fallback 可以保留。不要重复历史失败配置；不要仅改注释。',
        '认真读当前源码，历史已采纳改动已经在当前源码中，不要把它重复当成新优化。必须具体说明修改了哪个 kernel、分块或调用参数，以及哪些慢 case 会因此改善。',
        '硬件为 910B4-1，CANN9.0.0、Triton3.5.1、Flagtree0.6.0。当前Ascend后端NPUOptions保留num_warps与num_stages，910B默认num_stages=2；不能按CUDA warp语义套用调参规则，具体配置须完整编译和评测。编译器选项必须来自现有代码或知识。',
        '每次launch的grid必须在1到65535之内。大输入使用有界grid和grid-stride循环覆盖全部元素，不可直接按n/BLOCK启动百万program；越界launch可能只表现为profiler缺少事件和pandas.str错误。',
        '在@triton.jit内使用字面量、constexpr参数或tl.constexpr(value)，不能引用普通Python全局常量。必须有顶层def run或run指向实际定义的实现函数；不得返回占位符。',
        '工作量中的逻辑字节不是 HBM 物理字节。没有可信 roof 时不声称数值 Roofline。',
        '只能把 PipeUtilization 当作相应流水活跃比例；不将 tile 逻辑字节除以全kernel时间解释为实测L1/HBM带宽，也不把片上容量估算当作已编译证明。容量与性能推测必须由完整测试验证。',
        '程序会负责完整评测和保留最佳，你只提供有根据的可运行候选。',
        '算子：' + operator, '架构与既有经验：\n' + knowledge,
        '历史尝试：\n' + json.dumps(history, ensure_ascii=False),
        '当前逐case事实：\n' + json.dumps(facts, ensure_ascii=False),
        '当前最佳源码：\n' + source,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:19655')
    parser.add_argument('--claude', default='/data/hanle/ascend-optimization/runtime/kg-controller/claude')
    # 宿主包装器会把别名 deepseek-flash 映射到实际 DeepSeek 配置；直接传
    # deepseek-v4-flash[1m] 会被 Claude SDK 判为未知模型。
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--profile-cases', type=int, default=2)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--plain-json', action='store_true',
                        help='要求CC直接返回JSON文本，避免部分模型把结构化工具参数填成占位符')
    parser.add_argument('--thinking-budget', type=int,
                        help='向CC传入MAX_THINKING_TOKENS；供应商是否严格执行需看实际响应')
    parser.add_argument('--max-output-tokens', type=int,
                        help='向CC传入CLAUDE_CODE_MAX_OUTPUT_TOKENS，给思考和完整源码设置总输出预算')
    parser.add_argument('--replay', action='store_true',
                        help='以历史已验证候选复现方法；最终报告明确标注复现，不宣称重新发现')
    a = parser.parse_args()
    if a.thinking_budget is not None and a.thinking_budget < 1024:
        parser.error('--thinking-budget必须至少1024')
    if a.max_output_tokens is not None and a.max_output_tokens < 1024:
        parser.error('--max-output-tokens必须至少1024')
    if (a.thinking_budget is not None and a.max_output_tokens is not None
            and a.max_output_tokens <= a.thinking_budget):
        parser.error('总输出预算须大于思考预算，为完整源码保留空间')
    a.output.mkdir(parents=True, exist_ok=True)
    state_path = a.output / 'campaign-state.json'
    lock_path = a.output / 'campaign.lock'
    import fcntl
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
            'operator': a.operator, 'seed_sha256': digest(a.seed.read_text(encoding='utf-8')),
            'status': 'running', 'history': [], 'best': None}
        assert state['operator'] == a.operator and state['seed_sha256'] == digest(a.seed.read_text(encoding='utf-8'))
        knowledge = knowledge_for(a.operator, a.knowledge)
        configuration = {'server': a.server, 'model': a.model, 'claude': a.claude,
                         'profile_cases': a.profile_cases,
                         'knowledge_sha256': digest(knowledge)}
        if a.plain_json and ('configuration' not in state
                             or 'generation_format' in state['configuration']):
            configuration['generation_format'] = 'plain-json'
        for key in ('thinking_budget', 'max_output_tokens'):
            if getattr(a, key) is not None:
                configuration[key] = getattr(a, key)
        if a.replay:
            configuration['workflow_kind'] = 'historical-replay'
            state['workflow_kind'] = 'historical-replay'
        if 'configuration' in state and state['configuration'] != configuration:
            raise ValueError('恢复参数或知识文件已改变，请使用新批次')
        state['configuration'] = configuration
        if state['status'] == 'completed':
            assert digest((a.output / 'final.py').read_text(encoding='utf-8')) == state['final_source_sha256']
            print('CAMPAIGN_ALREADY_COMPLETED', a.operator, state['final_scores'], flush=True)
            return
        commit = subprocess.run(['git', '-C', str(PROJECT.parents[1]), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True).stdout.strip()
        if state.get('repo_commit', commit) != commit:
            raise ValueError('仓库 HEAD 已改变，停止恢复原批次')
        state['repo_commit'] = commit
        knowledge_snapshot = a.output / 'knowledge.md'
        if knowledge_snapshot.exists() and knowledge_snapshot.read_text(encoding='utf-8') != knowledge:
            raise ValueError('保存的知识快照与当前配置不一致')
        if not knowledge_snapshot.exists():
            knowledge_snapshot.write_text(knowledge, encoding='utf-8')
        state.update(status='running', current_error=None)
        save(state_path, state)
        contract = None
        if state['best']:
            contract, _ = checked(Path(state['seed_result']))
        else:
            seed_dir = a.output / 'seed-evaluation'
            seed_result = evaluate(a.operator, a.seed, seed_dir, 'seed', a.server)
            contract, score = checked(seed_result)
            state.update(seed_result=str(seed_result), best={'result': str(seed_result), 'score': score})
            save(state_path, state)
        for index in range(1, a.rounds + 1):
            if any(row['round'] == index for row in state['history']):
                continue
            directory = a.output / f'round-{index:02d}'
            directory.mkdir(exist_ok=True)
            best = Path(state['best']['result'])
            best_request = json.loads(best.with_name(best.name.replace('.result.json', '.request.json')).read_text(encoding='utf-8'))
            source = best_request['implementation']['sources'][0]['content']
            print('ROUND', index, 'BEST', state['best']['score'], flush=True)
            row = {'round': index, 'parent_result': str(best), 'parent_score': state['best']['score']}
            try:
                state.update(current_round=index, stage='profiling')
                save(state_path, state)
                report = profile_best(a.operator, best, directory / 'profiles', a.server, a.profile_cases)
                prompt = prompt_for(a.operator, source, report, state['history'], knowledge)
                (directory / '给CC的提示词.md').write_text(prompt, encoding='utf-8')
                if a.replay:
                    # 历史方法复现保留原候选和诊断，不要求模型重新发明源码。
                    row.update(status='REPLAYED', source_sha256=digest(source),
                               hypothesis='固化历史方法，采集新诊断并保留已验证源码；未启动模型生成。')
                    state['history'].append(row)
                    state.update(stage='replayed')
                    save(state_path, state)
                    print('REPLAYED', index, 'BEST', state['best']['score'], flush=True)
                    continue
                candidate = directory / 'candidate.py'
                if not candidate.exists():
                    schema = json.dumps({'type': 'object', 'properties': {key: {'type': 'string'}
                        for key in ('source', 'hypothesis', 'expected_effect')},
                        'required': ['source', 'hypothesis', 'expected_effect'], 'additionalProperties': False})
                    state.update(stage='generating')
                    save(state_path, state)
                    # 模型偶尔只返回占位符。保留原响应，最多一次格式修复，不消耗新优化轮。
                    for attempt in (1, 2):
                        response = directory / ('claude-response.json' if attempt == 1 else 'claude-repair-response.json')
                        if not response.exists():
                            command = [a.claude, '-p', '--model', a.model, '--output-format', 'stream-json', '--verbose', '--max-turns', '1',
                                 '--tools', '', '--strict-mcp-config', '--setting-sources', '',
                                 '--settings', '{"disableAllHooks":true}']
                            if not a.plain_json:
                                command += ['--json-schema', schema]
                            generation_env = os.environ.copy()
                            if a.thinking_budget is not None:
                                generation_env['MAX_THINKING_TOKENS'] = str(a.thinking_budget)
                            if a.max_output_tokens is not None:
                                generation_env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(a.max_output_tokens)
                            run(command,
                                response, prompt=prompt, timeout=1200, cwd=directory,
                                env=generation_env)
                        try:
                            proposal = parse_candidate(response.read_text(encoding='utf-8'))
                            break
                        except (ValueError, KeyError, SyntaxError, TypeError) as error:
                            if attempt == 2:
                                raise
                            prompt += '\n上次输出校验失败：' + str(error) + '\n必须输出包含 run 的完整源码；禁止占位符、截断源码或仅返回补丁。'
                            save(directory / 'generation-error.json', {'error': str(error)})
                    candidate.write_bytes(proposal['source'].encode())
                    save(directory / 'proposal.json', {k: v for k, v in proposal.items() if k != 'source'})
                row.update(json.loads((directory / 'proposal.json').read_text(encoding='utf-8')))
                row['source_sha256'] = digest(candidate.read_text(encoding='utf-8'))
                result_path = directory / 'evaluation/trial-1.result.json'
                state.update(stage='evaluating')
                save(state_path, state)
                if not result_path.exists():
                    evaluate(a.operator, candidate, directory / 'evaluation', 'trial', a.server)
                result, score = checked(result_path, contract)
                row.update(status='PASSED', score=score, result=str(result_path))
                if score > state['best']['score'] * 1.005:
                    state['best'] = {'result': str(result_path), 'score': score}
                    row['accepted'] = True
                else:
                    row['accepted'] = False
            except Exception as error:
                row.update(status='FAILED', error=str(error))
                if isinstance(error, InterruptedDeviceRequest):
                    state.update(status='needs_attention', current_error=str(error))
                    save(state_path, state)
                    raise
                failed_result = directory / 'evaluation/trial-1.result.json'
                if failed_result.exists():
                    failed = json.loads(failed_result.read_text(encoding='utf-8'))
                    row['evaluation_status'] = failed.get('status')
                    row['failure_detail'] = str(failed.get('log', failed.get('detail', '')))[-5000:]
                    row['failed_cases'] = [{key: workload.get(key) for key in ('uuid', 'status', 'log', 'abs_err', 'rel_err')}
                        for workload in failed.get('per_workload', []) if workload.get('status') != 'PASSED'][:3]
                if 'claude-response.json' in str(error):
                    try:
                        raw = (directory / 'claude-response.json').read_text(encoding='utf-8')
                        row['claude_error'] = raw[-2000:]
                    except OSError:
                        pass
                # 设备事故或评测超时应停下，不能向同卡追加任务。
                for port in (19655, 19657):
                    scheduler = call('http://127.0.0.1:' + str(port), '/status', timeout=30)['scheduler']
                    if scheduler['broken'] or scheduler['active'] or scheduler['waiting']:
                        state.update(status='needs_attention', current_error=str(error))
                        save(state_path, state)
                        raise RuntimeError('同卡服务仍忙或异常，停止 campaign') from error
            state['history'].append(row)
            save(state_path, state)
            print('RESULT', json.dumps(row, ensure_ascii=False), flush=True)
        final_dir = a.output / 'final-evaluation'
        state.update(stage='final_evaluation')
        save(state_path, state)
        best = Path(state['best']['result'])
        req = json.loads(best.with_name(best.name.replace('.result.json', '.request.json')).read_text(encoding='utf-8'))
        final_source = a.output / 'final.py'
        final_source.write_bytes(req['implementation']['sources'][0]['content'].encode())
        evaluate(a.operator, final_source, final_dir, 'final', a.server, repeats=2)
        final_scores = [checked(final_dir / f'final-{i}.result.json', contract)[1] for i in (1, 2)]
        state.update(status='completed', stage='completed', final_scores=final_scores,
                     final_score=min(final_scores), final_source_sha256=digest(final_source.read_text(encoding='utf-8')))
        save(state_path, state)
        print('CAMPAIGN_COMPLETED', json.dumps({'operator': a.operator, 'final_scores': final_scores}), flush=True)


if __name__ == '__main__':
    main()
