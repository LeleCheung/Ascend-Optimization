"""Record structural and evidence review for a model-authored centering repair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', default='ir-011')
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    workspace = run / 'workspace'
    assert read(call / 'submission.json')['status'] == 'awaiting_ir_review'
    assert (call / 'model-usage.json').exists()
    entry = read(workspace / 'ir-inventory.json')['entries'][0]
    original = call / 'input-ir' / entry['path']
    candidate = call / 'output-ir' / entry['path']
    a, b = original.read_text(), candidate.read_text()
    assert next(line for line in a.splitlines() if 'tt.func public' in line) == next(
        line for line in b.splitlines() if 'tt.func public' in line)
    assert a[a.index('      %w_acc:3'):a.index('      %w_cnt_f')] == b[
        b.index('      %w_acc:3'):b.index('      %w_cnt_f')]
    narrow_a = a[a.index('    %cnt_7:3'):a.index('    %0 = tt.addptr %mean_pointer')]
    narrow_b = b[b.index('    %cnt_7:3'):b.index('    %0 = tt.addptr %mean_pointer')]
    narrow_b = re.sub(r'^    %final_mean_res = .*\n', '', narrow_b, flags=re.M).replace(
        'scf.yield %final_mean_13, %final_mean_res, %var_20, %inv_std_21 : f32, f32, f32, f32',
        'scf.yield %final_mean_13, %var_20, %inv_std_21 : f32, f32, f32')
    assert narrow_a == narrow_b
    output_a = a[a.index('%var = tt.splat %stats_ws#0'):]
    output_b = b[b.index('%var = tt.splat %stats_ws#0'):].replace('%stats_ws#3', '%stats_ws#2')
    assert output_a == output_b
    uses, results = {}, {}
    for index, line in enumerate((call / 'agent-events.jsonl').read_text().splitlines()):
        event = json.loads(line)
        message = event.get('message', {})
        if not isinstance(message, dict):
            continue
        for block in message.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                uses[block['id']] = index, block
            if block.get('type') == 'tool_result':
                results[block['tool_use_id']] = index, block
    decisions = [(uses[key][0], index) for key, (index, result) in results.items()
        if key in uses and uses[key][1]['name'] == 'Bash'
        and str(uses[key][1]['input'].get('command', '')).strip().endswith(' decision')
        and 'decision_recorded' in str(result.get('content')) and not result.get('is_error')]
    assert len(decisions) == 1
    decision_use, decision_result = decisions[0]
    edits = [(index, use) for index, use in uses.values() if use['name'] == 'Edit'
        and str(use['input'].get('file_path', '')).endswith('.ttir')]
    assert edits and min(index for index, _ in edits) > decision_result
    reads = {}
    for token in ['context.json', 'feedback.json', 'profiling.json', 'hotspots.json',
            'timeline.json', 'frozen.py', '.ttir', '.ttadapter', '.bcmlir',
            'prior-ir010-numerics-original.json', 'prior-ir010-numerics-candidate.json', 'validation-policy.json']:
        matches = []
        for key, (index, use) in uses.items():
            if index >= decision_use or use['name'] not in ['Read', 'Bash'] or token not in str(use['input']) or key not in results:
                continue
            result_index, result = results[key]
            if result_index >= decision_use or result.get('is_error') or not result.get('content'):
                continue
            matches.append(dict(use_event=index, result_event=result_index,
                tool=use['name'], input=use['input']))
        assert matches, token
        reads[token] = matches
    save(call / 'evidence-read-review.json', dict(status='preedit_dual_and_actual_low_level_reads_verified',
        decision_use_event_index=decision_use, decision_result_event_index=decision_result,
        first_ir_edit_event_index=min(index for index, _ in edits), evidence_reads=reads))
    mechanism = 'Data-derived origin;center lane-mean reduction and wide normalization;retain absolute mean for output state'
    save(call / 'semantic-schedule-review.json', dict(status='structural_semantic_review_passed',
        abi_identical=True, statistics_welford_body_identical=True, narrow_statistics_equations_identical=True,
        narrow_output_identical_after_result_binding=True, mechanism=mechanism,
        real_arithmetic_identity='A=sum(count*(lane_mean-c))/N;absolute mean=c+A;output=((x-c)-A)*w*invstd+bias. Variance uses reconstructed absolute mean and existing merge formula.',
        range='Public wrapper for this EVEN specialization selects batch8 and positive spatial multiples128. Extra load addresses batch0/spatial0 of the current feature;original addresses and stores are preserved.',
        floating_point_limits='Reassociation changes rounding;finite-difference overflow and nonfinite propagation remain device-check concerns. No universal all-FP32 equivalence claim. Multi-step Welford remains in absolute coordinates.',
        input_ir_sha256=sha(original), output_ir_sha256=sha(candidate)))
    save(call / 'ir-review.json', dict(decision='approved', reviewed_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=read(run / 'config.json')['source_sha256'], changed_ir={entry['path']: sha(candidate)},
        mechanism=mechanism, comparison='Input is unretained ir010;qualified night ir004 is the CPU/formal comparator',
        claim_corrections=[
            'At128 the recorded1.52587890625e-5 error is1ULP;model final reply corrected its registered30ULP estimate.',
            'Statistics residualization occurs once after all m_steps;output preloads add17 tile subtracts at target,before backend optimization.',
            'A data-derived origin is expressible in Triton;fixed-source isolation motivates this IR experiment.',
            'MTE2 work absorption,UB peak,unchanged BAR and exact VSUB increments remain predictions.',
            'Finite-origin selection alone gives no global all-FP32 equivalence;fixed full contract and existing25 diagnostics decide qualification.'
        ], scope='Approved for CPU lowering and existing device gates;no performance or numerical-retention conclusion'))
    print(json.dumps(dict(status='host_review_passed', invocation=args.invocation)))


if __name__ == '__main__':
    main()
