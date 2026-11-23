"""Prepare model-only repair inputs from an explicitly unretained IR exploration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', default='ir-010')
    parser.add_argument('--next', default='ir-011')
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = run / 'workspace'
    assert not (run / 'agent-calls' / args.next).exists(), 'Model call already exists'
    # This record was explicitly created when the repair branch was selected.
    branch = read(run / ('exploration-parent-' + args.next + '.json'))
    assert branch['status'] == 'explicit_unretained_exploration_repair'
    assert branch['previous_invocation'] == args.previous
    entries = read(workspace / 'ir-inventory.json')['entries']
    assert len(entries) == 1
    assert sha(workspace / entries[0]['path']) == branch['input_ir_sha256']
    assert sha(workspace / 'frozen.py') == branch['source_sha256']
    lowered = {}
    for path in (run / 'evaluations' / args.previous / 'compiler-receipts').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get('override_applied'):
                continue
            for artifact in row.get('artifacts', []):
                if artifact['kind'] not in {'ttadapter', 'bcmlir', 'npubin'}:
                    continue
                key = (row['ast_sha256'], artifact['kind'])
                assert key not in lowered or lowered[key]['sha256'] == artifact['sha256']
                lowered[key] = dict(artifact, ast_sha256=row['ast_sha256'], symbol=row['symbol'])
    assert len(lowered) == 3, 'Repair input must include the exact compiled target'
    inventory = []
    for artifact in lowered.values():
        original = Path(artifact['path'])
        assert sha(original) == artifact['sha256']
        relative = Path('compiled-readonly') / artifact['ast_sha256'] / original.name
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        inventory.append(dict(artifact, path=relative.as_posix(), original=str(original)))
    save(workspace / 'lowered-inventory.json', dict(source_sha256=branch['source_sha256'],
        input_status='unretained_exploration_repair', input_invocation=args.previous, entries=inventory))
    inputs = {
        'formal': run / 'evaluations' / args.previous / 'result.json',
        'supplements': run / 'evaluations' / args.previous / 'supplementary-summary.json',
        'retention': run / 'retention-ir010.json',
        'post-mechanism': run / 'post-mechanism-ir010.json',
        'post-facts': run / 'post-mechanism-ir010-facts.json',
        'numerics': run / 'agent-calls' / args.previous / 'supplemental-grouping/summary.json',
        'numerics-protocol': run / 'agent-calls' / args.previous / 'supplemental-grouping/protocol.json',
        'numerics-original': run / 'agent-calls' / args.previous / 'supplemental-grouping/original/result.json',
        'numerics-candidate': run / 'agent-calls' / args.previous / 'supplemental-grouping/candidate/result.json',
    }
    for name, original in inputs.items():
        shutil.copyfile(original, workspace / ('prior-ir010-' + name + '.json'))
    for destination in [run / 'validation-policy.json', workspace / 'validation-policy.json']:
        shutil.copyfile(args.policy, destination)
    context = read(workspace / 'context.json')
    context.update(input_status=branch['status'], exploration_parent=branch,
        goal='Repair the unretained numeric regression; qualified night ir004 remains the performance comparator')
    save(workspace / 'context.json', context)
    save(workspace / 'feedback.json', dict(status='numeric_regression_repair_required',
        parent=read(inputs['retention']), complete_formal=read(inputs['formal']),
        supplements=read(inputs['supplements']), numerical_diagnostic=read(inputs['numerics']),
        notes='The input is ir010, not the qualified best. Repair the observed regression using the frozen diagnostics.'))
    save(workspace / 'codex-brief.json', dict(
        role='Codex evidence analysis; CC/DeepSeek chooses and authors every candidate IR edit',
        findings=[
            'Statistics256 reduced recurrence steps and retained an observed target benefit, but a newly failing offset128/S256 output blocks retention.',
            'Original and candidate both pass19/25 diagnostics. Five shared offset4096 failures are inherited limits; the candidate also recovers original S512/offset128.',
            'Output normalization subtracts an absolute FP32 mean. Investigate rounding and cancellation under large offsets and small variance using actual TTIR and frozen numerical outputs.',
            'Post simulator has38/38 independent PC clock matches; MTE2 copies100to68,BAR484to330,DIV82to58,reads268to272KiB/program. Partial windows and child139 remain recorded.'
        ],
        direction='Investigate a general stable centering representation, such as a common origin with a residual mean through statistics and normalization, while producing the required absolute mean and running state. Choose the mechanism from the evidence; no candidate code is supplied.',
        constraints=[
            'Read context, feedback, current dual evidence, current compiled artifacts, and frozen numerical diagnostics before registering the repair.',
            'Keep the source, ABI, compiler options and original fallback semantics. A general numerical repair may adjust the related statistics/normalization IR; preregister its full scope.',
            'Avoid guards fitting the observed test shape or offsets. Preserve mathematical outputs and original tolerances. The qualified comparator remains night ir00439.261332us.',
            'Use fixed27 complete cases, existing dual36 supplements and the already frozen25 numerical diagnostics. Require the new regression repaired and no new failures relative to the qualified original; inherited limits stay explicit.',
            'ABBA only for concrete measurement concerns. Hashes are automatically associated by tools. Record expected instruction/wait/resource costs and replay the candidate after compiling.'
        ], open_search=True))
    save(run / ('numeric-repair-preparation-' + args.next + '.json'),
        dict(status='ready_for_model', prepared_at=datetime.now(timezone.utc).isoformat(),
             branch=branch, current_lowered_artifacts=inventory,
             validation_policy=read(args.policy), operator_implementation_edits=0))
    print(json.dumps(dict(status='ready_for_model', invocation=args.next)))


if __name__ == '__main__':
    main()
