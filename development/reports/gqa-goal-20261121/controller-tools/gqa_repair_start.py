"""Return complete source-bound GQA failures to the model for a recorded repair."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--host-analysis', type=Path)
    parser.add_argument('--review-feedback', type=Path, help='Explicit source review failure, before device evaluation')
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = run / 'workspace'
    source = workspace / 'candidate.py'
    result_path = run / 'iterations' / args.previous / 'parent-validation/results.json'
    review_feedback = read(args.review_feedback) if args.review_feedback else None
    if review_feedback:
        assert review_feedback['source_sha256'] == sha(source)
        assert review_feedback['status'] == 'source_review_repair_required'
        result = dict(source_sha256=sha(source), cases=[])
    else:
        result = read(result_path)
        assert result['source_sha256'] == sha(source)
        assert len(result['cases']) == 4
        assert not all(row.get('candidate_correctness', {}).get('official_checker', {}).get('passed')
            for row in result['cases'])
    assert not (run / 'agent-calls' / args.invocation).exists()
    config = read(run / 'config.json')
    save(run / ('config-before-' + args.invocation + '.json'), config)
    config.pop('required_simulator_evidence', None)
    config['evidence_mode'] = 'repair-feedback'
    save(run / 'config.json', config)
    failures = []
    for row in result['cases']:
        error = row.get('candidate_error') or row.get('error') or ''
        resource = re.search(r'requires (\d+) bits while (\d+) bits available', error)
        resources = [dict(resource=kind, required_bytes=int(required) // 8,
                          available_bytes=int(available) // 8)
                     for kind, required, available in dict.fromkeys(re.findall(
                         r'(\w+) overflow, requires (\d+) bits while (\d+) bits available', error))]
        failures.append(dict(case_id=row['case_id'], error=error,
            resource_bytes=dict(required=int(resource[1]) // 8, available=int(resource[2]) // 8)
                if resource else None,
            resource_errors=resources,
            official_checker=row.get('candidate_correctness', {}).get('official_checker'),
            candidate_timing=row.get('candidate_timing'), native_timing=row.get('native_timing')))
    save(workspace / 'repair-feedback.json', dict(source_sha256=sha(source),
        feedback_kind='source_review' if review_feedback else 'formal_evaluation',
        result=str(args.review_feedback if review_feedback else result_path),
        result_sha256=sha(args.review_feedback if review_feedback else result_path), cases=failures,
        source_review=review_feedback, complete_formal_result=None if review_feedback else result,
        host_correction=('源代码审查发现适用范围问题，当前候选尚未设备评测；修复后沿原四项验收。'
            if review_feedback else '实际preflight只交接源码审查；编译、资源与正确性结果由本次四项执行给出。')
            + '此前双分析按原source_sha256保留为历史背景。'))
    folder = workspace / 'failed-ir' / args.previous
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = []
    cache = result_path.parent / 'cache'
    for path in sorted(cache.rglob('*')):
        if path.is_file() and path.suffix in {'.ttir', '.bcmlir', '.ttadapter'}:
            target = folder / (str(len(artifacts)) + '-' + path.name)
            if target.exists():
                assert sha(target) == sha(path), 'Previously frozen failed IR changed'
            shutil.copyfile(path, target)
            artifacts.append(dict(path=target.relative_to(workspace).as_posix(), sha256=sha(path)))
    save(workspace / 'context-brief.json', dict(source_sha256=sha(source), phase='explicit_unretained_repair',
        feedback='repair-feedback.json', actual_failed_ir=artifacts,
        current_profiling=None, current_simulator=None,
        preserved_best_source_sha256=read(run / (
            'selection.json' if (run / 'selection.json').exists() else 'retained-parent.json'))['source_sha256'],
        historical_evidence='current-profiling.json与simulator-evidence.json保留原版身份，当前失败候选尚无设备重放。'))
    prompt = '''修复当前GQA未保留候选。读取task.json、contract.json、context-brief.json、repair-feedback.json、candidate.py及必要的实际失败IR。四项原合同结果已全部返回；先核对每项真实编译、资源或数值错误。保持四项checker、golden、dtype、计时口径及当前已验收精度路径。请选择一个针对实际失败原因的具体修复机制，明确其对精度、活跃缓冲、交接次数和并行度的取舍。原保留版独立冻结，修复沿当前失败源码明确接续。
原版profiling和模拟器仅为历史背景。本次决策以当前失败反馈和实际IR为依据，evidence使用evaluation/source_ir类别；无需重读原版模拟器，不将其标为当前候选观测。不要声称当前源码已通过编译或存在真机收益。用Edit写diagnosis.json，包含selected_layer、layer_reason、simulator_incremental_judgment及证据、假设、改法、预测、验证与未知项，调用工具桥decision --record diagnosis.json后才编辑candidate.py。全部候选修复由你完成，host仅分配、元数据和启动，设备计算严格Triton。工具桥preflight返回awaiting_source_review后保持字节并交接。无总轮数、工具或时间预算，完整四项反馈后继续研究。
'''
    if review_feedback:
        prompt = prompt.replace('四项原合同结果已全部返回；先核对每项真实编译、资源或数值错误。',
            '当前候选尚未设备评测，源代码审查反馈已返回；先核对真实地址、分支与适用条件问题，保留该验收阶段。')
        prompt = prompt.replace('及必要的实际失败IR', '及审查引用的相关IR')
    if args.host_analysis:
        analysis = read(args.host_analysis)
        assert analysis['source_sha256'] == sha(source)
        save(workspace / 'host-repair-analysis.json', analysis)
        prompt += '\n修改前还需读取host-repair-analysis.json，分别记录宿主建议和你的具体修复。\n'
    prompt_path = run / (args.invocation + '-prompt.md')
    prompt_path.write_text(prompt, encoding='utf8')
    command = [sys.executable, '-m', 'hgkda.runtime.agent_runner', '--config', str(run / 'config.json'),
        '--model-settings', str(run / 'model-settings.json'), '--invocation', args.invocation,
        '--prompt-file', str(prompt_path)]
    with (run / ('generation-' + args.invocation + '.log')).open('x') as stream:
        process = subprocess.Popen(command, env=dict(os.environ, PYTHONPATH=str(Path(config['package']) / 'src')),
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, command=command, source_sha256=sha(source),
        parent_invocation=args.previous, branch='explicit_unretained_repair',
        started_at=datetime.now(timezone.utc).isoformat())
    save(run / ('process-' + args.invocation + '.json'), record)
    print(record)


if __name__ == '__main__':
    main()
