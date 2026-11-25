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
    prompt = """Read-only result handoff for the same overnight study. Read complete-results.json fully: three GroupNorm versions, three writeback versions, and original/failed/repaired GQA results with retained choices. Only Read is available; do not edit candidate, diagnosis, or scheduling files.
Confirm receipt concisely in English. List original/modified target-shape timings for the two accepted IR cases and full-contract geometric-mean timing changes; identify writeback ir-003 as a negative result against retained ir-002. Verify repaired GQA official checkers and timings for all four cases; calculate case4 improvement against the retained original. Explain restored causal tiling for case1 resource repair and wider KV tiling for the three noncausal cases. Failed GQA improved case4 but failed case1 compilation; repaired GQA fully passed. Post-change simulation is pending; verify mechanisms with later replay. Preserve physical-UB estimate and partial-window scope without extrapolating to all 20 cases. No implementation edits, new tests, ABBA, or search budgets. Optionally state one separately labeled research hypothesis."""
    if args.gqa_only:
        prompt = """Read-only full-result handoff for GQA continuation. Read complete-results.json fully, including original four-case baseline, every new candidate, post-collection status, and retained selection. Only Read; edit nothing.
Confirm receipt of full four-case results and list actual case4 timings/checker states. Read the starting point from parent results and calculate retained-target improvement; also report all-four behavior and negative results. Separate actual source/IR layer and partial-replay scope; full 20-case official acceptance remains future work. Distinguish resource estimates, bitwise-equivalence/movement predictions from actual compiler/checker/replay outcomes. Optionally state one hypothesis as a suggestion; this invocation only acknowledges full results."""
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
