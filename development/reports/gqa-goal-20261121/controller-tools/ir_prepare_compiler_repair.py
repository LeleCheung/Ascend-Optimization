"""Prepare an explicit model repair branch from preserved, uncompiled IR bytes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--next', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = run / 'workspace'
    previous = run / 'agent-calls' / args.previous
    assert (previous / 'model-usage.json').is_file()
    assert not (run / 'agent-calls' / args.next).exists()
    lowering = read(previous / 'lowering-review.json')
    assert lowering['status'] == 'lowering_review_failed'
    inventory = read(workspace / 'ir-inventory.json')
    assert sha(workspace / 'frozen.py') == inventory['source_sha256']
    files = []
    for entry in inventory['entries']:
        current = workspace / entry['path']
        submitted = previous / 'output-ir' / entry['path']
        assert sha(current) == sha(submitted), 'Preserve the exact failed model submission'
        files.append(dict(path=entry['path'], input_sha256=sha(current),
                          submitted=str(submitted), qualified_ir_sha256=entry['original_sha256']))
    copied = []
    for name, original in [('lowering', previous / 'lowering-review.json'),
                           ('compiler-log', run / ('lowering-' + args.previous + '.log')),
                           ('registered-decision', previous / 'before-edit-decision.json')]:
        target = workspace / ('prior-' + args.previous + '-' + name + original.suffix)
        assert not target.exists()
        shutil.copyfile(original, target)
        copied.append(dict(path=target.name, sha256=sha(target)))
    branch = dict(status='explicit_unretained_exploration_repair',
                  previous_invocation=args.previous, next_invocation=args.next,
                  source_sha256=inventory['source_sha256'], ir_inputs=files,
                  scope='Model-only compilation repair; no target implementation edits by host')
    context = read(workspace / 'context.json')
    context.update(input_status=branch['status'], exploration_parent=branch,
                   goal='Repair the preserved compiler failure, then compare against the qualified unchanged IR')
    save(workspace / 'context.json', context)
    save(workspace / 'feedback.json', dict(status='compiler_repair_required',
         parent=branch, lowering=lowering, evidence=copied,
         requested_action='Read the exact compiler diagnostics and original dialect examples; repair syntax within the registered mechanism. Re-register before editing. Hand back a different mechanism for a new branch.',
         validation='CPU parsing and lowering first; fixed complete contract and existing supplements only after comparator verification; post simulator replay follows'))
    save(run / ('exploration-parent-' + args.next + '.json'), dict(branch,
         prepared_at=datetime.now(timezone.utc).isoformat(),
         preparation_tool_sha256=sha(Path(__file__)), copied_feedback=copied,
         operator_implementation_edits=0))
    print(json.dumps(dict(status='ready_for_model_repair', invocation=args.next)))


if __name__ == '__main__':
    main()
