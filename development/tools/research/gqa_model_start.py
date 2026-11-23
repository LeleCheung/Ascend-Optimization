"""Provide frozen GQA evidence to the existing unbounded model runner."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--host-analysis', type=Path)
    parser.add_argument('--observation', type=Path, help='Immutable qualified observation for a later call')
    parser.add_argument('--compact-evidence', action='store_true')
    parser.add_argument('--evidence-mode', choices=['simulator', 'profiling-source-ir'], default='simulator')
    args = parser.parse_args()
    run, template = args.run.resolve(), args.template.resolve()
    observation = args.observation.resolve() if args.observation else run
    workspace = run / 'workspace'
    source = workspace / 'candidate.py'
    digest = sha(source)
    state = read(observation / 'observation-status.json')
    include_simulator = args.evidence_mode == 'simulator'
    if state['status'] == 'parent_dual_evidence_ready':
        assert state['source_sha256'] == digest
    else:
        # Generation using complete hardware/IR evidence may overlap its parent's
        # immutable post replay. Retention still requires the replay binding.
        assert not include_simulator
        assert state['status'] in {'collecting_pair', 'collecting_simulator'}
        assert sha(observation / 'workspace/candidate.py') == digest
        formal = read(observation / 'parent-validation/results.json')
        assert formal['source_sha256'] == digest
        assert formal['contract_sha256'] == sha(run / 'contract.json')
        assert {row['case_id'] for row in formal['cases']} == {1, 2, 4, 7}
        assert all(row['status'] == 'measured' and all(
            row[side + '_correctness']['official_checker']['passed']
            for side in ['candidate', 'native']) for row in formal['cases'])
        assert (observation / 'paired-parent/comparison.json').is_file()
        manifest = read(observation / 'capture-parent/candidate/actual-ir/manifest.json')
        assert manifest['source_sha256'] == digest
        assert manifest['contract_sha256'] == sha(run / 'contract.json')
    assert not (run / 'agent-calls' / args.invocation).exists()
    assert not (run / ('process-' + args.invocation + '.json')).exists()
    config = read(run / 'config.json')
    template_config = read(template / 'config.json')
    package = Path(template_config['package'])
    for name in ['simulator-evidence.json', 'simulator-evidence-full.json',
                 'current-profiling.json', 'current-profiling-full.json',
                 'previous-result.json', 'context-brief.json', 'diagnosis.json',
                 'repair-feedback.json', 'host-repair-analysis.json']:
        if (workspace / name).exists():
            history = run / 'evidence-history' / args.invocation
            history.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace / name, history / name)
            if name.startswith('simulator-evidence') or name in {
                    'diagnosis.json', 'repair-feedback.json', 'host-repair-analysis.json'}:
                (workspace / name).unlink()
    if not (workspace / 'skills').exists():
        shutil.copytree(package / 'skills', workspace / 'skills')
    shutil.copyfile(source, workspace / 'starter.py')
    shutil.copyfile(run / 'contract.json', workspace / 'contract.json')
    shutil.copyfile(template / 'model-settings.json', run / 'model-settings.json')
    shutil.copyfile(Path(__file__).with_name('contrast_decision_bridge.py'), run / 'decision_bridge.py')
    selections = [read(path) for path in [run / 'parent-selection.json',
                  *sorted(run.glob('retention-*.json'))] if path.exists()]
    prior_retained = any(row.get('source_sha256') == digest and row.get('retained', True)
                         for row in selections)
    save(workspace / 'starter-provenance.json', dict(
        kind='frozen_retained_gqa' if prior_retained else 'qualified_gqa_exploration',
        source_sha256=digest, prior_best_source_provided=prior_retained,
        origin=str(observation / 'retained-parent.json'), observation=str(observation)))
    save(workspace / 'knowledge.json', dict(records=[], scope='当前源码与实际测量证据；按记录保留历史两项残差精度结构。'))
    if include_simulator:
        original = read(observation / 'simulator-parent/summary.json')
        assert original['source_sha256'] == digest and original['reports']
        assert all(r['absolute_unit_validation']['status'] == 'verified' for r in original['reports'])
        simulator = dict(original, reported_status=original['status'], status='qualified_observation',
            capture_scope='各核的公共时钟观测窗口分别归一化，application_completed记录应用完成状态；真机完整调用独立计时。')
        save(workspace / 'simulator-evidence.json', simulator)
    save(workspace / 'current-profiling.json', dict(source_sha256=digest,
        comparison=read(observation / 'paired-parent/comparison.json')))
    if args.compact_evidence:
        from gqa_evidence_brief import compact
        compact(workspace, include_simulator=include_simulator)
    save(workspace / 'previous-result.json', dict(source_sha256=digest,
        current=read(observation / 'parent-validation/results.json'), historical=read(run / 'parent-results.json'),
        previous_iterations={path.parent.parent.name: read(path) for path in sorted(
            (run / 'iterations').glob('*/parent-validation/results.json'))}))
    contract = read(run / 'contract.json')
    result = read(observation / 'parent-validation/results.json')
    target_us = next(r['candidate_timing']['full_call_us'] for r in result['cases'] if r['case_id'] == 4)
    selection_path = run / 'selection.json'
    best_us = read(selection_path)['target_latency_us'] if selection_path.exists() else target_us
    host_analysis = ('结合当前热点、各流水窗口与实际IR选择下一项具体机制；'
                     '分别核对交接、访存、计算和缓冲资源的代价。')
    requested_mechanism = None
    if args.host_analysis:
        analysis = read(args.host_analysis)
        assert analysis['source_sha256'] == digest
        host_analysis = analysis['analysis']
        requested_mechanism = analysis.get('requested_mechanism')
    shutil.copyfile(Path(contract['golden']), workspace / 'official-golden.py')
    ir = workspace / 'actual-ir'
    if ir.exists():
        prior_ir = run / 'evidence-history' / args.invocation / 'previous-actual-ir'
        prior_ir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ir), str(prior_ir))
    ir.mkdir()
    artifacts = []
    for path in sorted((observation / 'capture-parent/candidate/actual-ir').rglob('*')):
        if path.is_file() and path.suffix in {'.ttir', '.bcmlir', '.ttadapter'}:
            target = ir / (str(len(artifacts)) + '-' + path.name)
            shutil.copyfile(path, target)
            artifacts.append(dict(path=target.relative_to(workspace).as_posix(), sha256=sha(path)))
    assert artifacts
    save(workspace / 'context-brief.json', dict(source_sha256=digest, actual_ir=artifacts,
        current_profiling='current-profiling.json',
        current_simulator='simulator-evidence.json' if include_simulator else None,
        evidence_mode=args.evidence_mode,
        prior_host_simulator_knowledge=True,
        previous_result='previous-result.json', current_target_us=target_us,
        retained_best_target_us=best_us, observation_parent_was_retained=prior_retained,
        parent_post_observation_status=state['status'],
        scope='固定case1/2/4/7，官方checker与golden身份沿原合同；原始对照与本轮续接独立。',
        host_analysis=host_analysis))
    config.update(package=str(package), paired_directory=str(observation / 'paired-parent'),
        active_source='candidate.py', context_kind='performance_continuation',
        require_preflight_review=True, formal_cases=4, optimization_policy='performance_first',
        method_manifest=template_config['method_manifest'], agent_tool_path=str(run / 'decision_bridge.py'),
        job=dict(operator='gqa', checks=dict(supplementary=dict(previous_total=4))),
        evidence_mode=args.evidence_mode)
    config.pop('required_simulator_evidence', None)
    if include_simulator:
        config['required_simulator_evidence'] = dict(path='simulator-evidence.json',
            sha256=sha(workspace / 'simulator-evidence.json'))
    save(run / 'config.json', config)
    save(run / 'controller-state.json', dict(status='prepared'))
    prompt = f'''继续优化CANNBench Level4 GQA。当前观察父版本{target_us:.5f}μs，保留最佳{best_us:.5f}μs；父版本身份见starter-provenance.json。先读取skills/hgkda/SKILL.md、task.json、contract.json、context-brief.json、previous-result.json、current-profiling.json、simulator-evidence.json、candidate.py及相关实际IR。固定case1/2/4/7、原checker、golden、输入输出dtype和计时合同；目标case4耗时低于保留最佳{best_us:.5f}μs，其他三项真实变化同时保存。当前概率计算结构沿父版本；调整中间dtype或计算顺序须事前明确数值风险与适用输入范围，由原完整四项checker验收，精度阈值保持。
模拟器为三个核的部分窗口，Cube和Vector各自归一化；明确热点和流水各自增加的判断，保留全调用completion缺失。选一个具体机制。Triton可表达的分块、布局和计算组织优先修改源码；发现单算子生成缺陷时在diagnosis中交接IR建议，本调用仅允许修改candidate.py与diagnosis.json。
修改前用Edit填写diagnosis.json，字段包含source_sha256、evidence(path/category/role/finding)、hypothesis、alternatives、discriminating_observation、change、expected、validation、selected_layer、layer_reason、simulator_incremental_judgment。证据路径使用workspace/文件名，category按paired_profiling/simulator/source_ir/evaluation分类。实际成功读取当前模拟器后，调用桥decision --record diagnosis.json登记，再改源码。仅用Edit编辑候选；Bash只调用工具桥。设备计算全部严格Triton，host仅分配、元数据和启动；禁止按case_id或输入值选路、输出缓存和外部后端计算。公共入口与原候选保持。
实现一个可评测机制后调用preflight --source candidate.py；收到awaiting_source_review后保持字节并交接宿主只读审查及完整四项真机评测。完整结果与修改后重放随后反馈。生成不设总轮数、工具或时间预算，此处交接属于验收边界。新增诊断和注释使用专业中文，简明记录实际收益与代价。
'''
    if not include_simulator:
        prompt = prompt.replace('、simulator-evidence.json', '')
        prompt = prompt.replace(
            '模拟器为三个核的部分窗口，Cube和Vector各自归一化；明确热点和流水各自增加的判断，保留全调用completion缺失。',
            '本调用提供当前成对profiling与实际IR，当前假设由这些材料及源码分析提出。宿主保留此前模拟器知识，此轮属于实际续接；工具增量仅按本轮实际输入记录。修改后仍开展模拟器重放。')
        prompt = prompt.replace(
            '实际成功读取当前模拟器后，调用桥decision --record diagnosis.json登记，再改源码。',
            '实际成功读取当前profiling、相关TTIR与BCMLIR后，调用桥decision --record diagnosis.json登记，再改源码。simulator_incremental_judgment明确当前剩余歧义及本轮无需新增模拟器输入的理由；evidence仅引用本轮实际提供和读取的材料。')
    if requested_mechanism:
        prompt += '\n本调用由宿主明确指定的独立实验：' + requested_mechanism + '\n'
    prompt_path = run / (args.invocation + '-prompt.md')
    prompt_path.write_text(prompt, encoding='utf8')
    command = [sys.executable, '-m', 'hgkda.runtime.agent_runner', '--config', str(run / 'config.json'),
        '--model-settings', str(run / 'model-settings.json'), '--invocation', args.invocation,
        '--prompt-file', str(prompt_path)]
    with (run / ('generation-' + args.invocation + '.log')).open('x') as stream:
        process = subprocess.Popen(command, env=dict(os.environ, PYTHONPATH=str(package / 'src')),
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, command=command, starting_source_sha256=digest,
        started_at=datetime.now(timezone.utc).isoformat())
    save(run / ('process-' + args.invocation + '.json'), record)
    print(record)


if __name__ == '__main__':
    main()
