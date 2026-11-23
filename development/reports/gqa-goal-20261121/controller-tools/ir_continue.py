"""Start an isolated IR stage from a qualified retained source/IR combination."""
import argparse
import copy
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_retained(run, previous_invocation, next_invocation):
    """Copy qualified parent bytes before a new hypothesis; author no IR edits."""
    workspace = run / 'workspace'
    previous = run / 'agent-calls' / previous_invocation
    assert (previous / 'model-usage.json').exists(), 'Wait for the previous model handoff'
    assert not (run / 'agent-calls' / next_invocation).exists()
    inventory = read(workspace / 'ir-inventory.json')
    assert sha(workspace / 'frozen.py') == inventory['source_sha256']
    plan = []
    for item in inventory['entries']:
        parent = Path(item['original'])
        current = workspace / item['path']
        captured = previous / 'output-ir' / item['path']
        assert sha(parent) == item['original_sha256'], 'Retained parent IR identity changed'
        assert sha(current) == sha(captured), 'Workspace differs from the previous model handoff'
        plan.append(dict(path=item['path'], previous_sha256=sha(current),
                         retained_sha256=sha(parent), retained_path=str(parent)))
    record = run / ('workspace-restore-before-' + next_invocation + '.json')
    assert not record.exists()
    save(record, dict(timestamp=datetime.now(timezone.utc).isoformat(), status='restoring',
        previous_invocation=previous_invocation, next_invocation=next_invocation, files=plan,
        scope='Exact qualified parent IR byte copies before the next model hypothesis; no authored transformation'))
    for item in plan:
        shutil.copyfile(item['retained_path'], workspace / item['path'])
        assert sha(workspace / item['path']) == item['retained_sha256']
    document = read(record)
    document['status'] = 'retained_ir_restored'
    save(record, document)
    return document


def prepare(parent, invocation, run, allow_unobserved=False):
    assert not run.exists(), 'A continuation stage requires a fresh directory'
    evaluation = parent / 'evaluations' / invocation
    result = read(evaluation / 'result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    source_manifest = read(evaluation / 'implementation/ir-manifest.json')
    if any(r['sha256'] != r['original_sha256'] for r in source_manifest['overrides']):
        checks = read(evaluation / 'supplementary-summary.json')
        assert checks and all(r['status'] == 'passed' for r in checks.values())
        selection = read(parent / 'selection.json')
        assert selection['status'] == 'retained_qualified_ir_improvement'
        assert selection['invocation'] == invocation
        assert selection['formal_result_sha256'] == sha(evaluation / 'result.json')
    observation = parent / 'observations' / invocation
    summary_path = observation / 'simulator/ir-summary.json'
    simulator = read(summary_path) if summary_path.exists() else None
    if simulator is None:
        assert allow_unobserved and invocation == 'unchanged-ir', 'Bound simulator evidence is required before model authoring'
        assert all(r['sha256'] == r['original_sha256'] for r in source_manifest['overrides'])
    else:
        assert simulator['source_sha256'] == source_manifest['source_sha256']
    workspace = run / 'workspace'
    workspace.mkdir(parents=True)
    config = copy.deepcopy(read(parent / 'config.json'))
    config.update(run_directory=str(run), frozen_source=str(workspace / 'frozen.py'),
                  continuation_parent=str(parent), continuation_invocation=invocation,
                  comparison_kind='retained_source_ir',
                  tools=str(run.parent / 'tools'))
    save(run / 'config.json', config)
    shutil.copyfile(parent / 'model-settings.json', run / 'model-settings.json')
    shutil.copyfile(evaluation / 'implementation/frozen.py', workspace / 'frozen.py')
    assert sha(workspace / 'frozen.py') == config['source_sha256']
    entries = []
    for item in source_manifest['overrides']:
        original = evaluation / 'implementation' / item['path']
        assert sha(original) == item['sha256']
        fixed = run / 'parent-inputs' / item['path']
        editable = workspace / item['path']
        for target in [fixed, editable]:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        entry = copy.deepcopy(item)
        entry.update(original=str(fixed), original_sha256=item['sha256'],
                     compiler_original_sha256=item['original_sha256'],
                     retained_parent_ir=str(original))
        entries.append(entry)
    save(workspace / 'ir-inventory.json', dict(source_sha256=config['source_sha256'],
         compiler_identity=source_manifest['compiler_identity'], entries=entries,
         origin='Qualified retained IR input; compiler-origin hashes separately preserved'))
    lowered = {}
    for receipt in (evaluation / 'compiler-receipts').glob('*.jsonl'):
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            if not row.get('override_applied'):
                continue
            for item in row['artifacts']:
                if item['kind'] in {'ttadapter', 'bcmlir', 'npubin'}:
                    key = row['ast_sha256'], item['kind']
                    assert key not in lowered or lowered[key]['sha256'] == item['sha256']
                    lowered[key] = dict(item, ast_sha256=key[0], symbol=row['symbol'])
    records = []
    for item in lowered.values():
        original = Path(item['path'])
        assert sha(original) == item['sha256']
        relative = Path('compiled-readonly') / item['ast_sha256'] / original.name
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        records.append(dict(item, path=relative.as_posix(), original=str(original)))
    assert {r['ast_sha256'] for r in entries} <= {r['ast_sha256'] for r in records}
    save(workspace / 'lowered-inventory.json', dict(source_sha256=config['source_sha256'], entries=records))
    if simulator is not None:
        shutil.copyfile(observation / 'paired/comparison.json', workspace / 'profiling.json')
        shutil.copyfile(summary_path, workspace / 'hotspots.json')
        save(workspace / 'timeline.json', dict(source_sha256=config['source_sha256'],
             reports=[dict(core=r['core'], timeline=r['timeline'],
                           absolute_unit_validation=r['absolute_unit_validation']) for r in simulator['reports']],
             original=str(summary_path), original_sha256=sha(summary_path)))
    target_case = config['experiment_goal']['target_case']
    target = next(r for r in result['per_workload'] if r['uuid'] == target_case and r['phase'] == 'timing')
    save(workspace / 'context.json', dict(operator=config['operator'], source_sha256=config['source_sha256'],
         target_case=target_case, candidate_layer='TTIR', source_edit_allowed=False,
         comparator=dict(parent_evaluation=str(evaluation), target=target),
         evidence_ready=simulator is not None,
         goal='Continue simulator replay from the qualified retained source/IR combination'))
    save(workspace / 'diagnosis.json', dict(source_sha256=config['source_sha256'],
         selected_layer='single_operator_ir', evidence=[], hypothesis='', change='', expected='', validation=''))
    save(workspace / 'feedback.json', dict(parent_result=result,
         parent_supplements=read(evaluation / 'supplementary-summary.json'),
         rule='Start from retained IR; preserve prior gains; compare new candidate against this parent'))
    save(run / 'continuation-provenance.json', dict(timestamp=datetime.now(timezone.utc).isoformat(),
         preparation_tool_sha256=sha(Path(__file__)),
         parent=str(parent), invocation=invocation, result_sha256=sha(evaluation / 'result.json'),
         manifest_sha256=sha(evaluation / 'implementation/ir-manifest.json'),
         source_sha256=config['source_sha256'],
         simulator_sha256=sha(summary_path) if simulator is not None else None,
         scope='Qualified parent IR byte copies; bound observations copied when available, otherwise collect before authoring'))
    save(run / 'status.json', dict(status='retained_ir_continuation_ready' if simulator is not None
                                  else 'awaiting_baseline_evidence', parent=invocation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--allow-unobserved', action='store_true', help='Prepare original-IR study only; collect bound evidence before authoring')
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.invocation, args.run.resolve(), args.allow_unobserved)
