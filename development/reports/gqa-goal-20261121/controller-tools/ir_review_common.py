"""Collect binding and pre-edit evidence facts; semantic approval stays with host."""
import argparse
import difflib
from pathlib import Path

from ir_campaign import read, save, sha
from ir_model import checked


def review(run, invocation):
    call = run / 'agent-calls' / invocation
    assert (call / 'model-usage.json').is_file(), 'Wait for the complete model handoff'
    assert read(call / 'submission.json')['status'] == 'awaiting_ir_review'
    workspace, _, initial, actual, editable = checked(run, invocation)
    stage = read(workspace / 'context.json').get('editable_ir_stage', 'ttir')
    assert stage in {'ttir', 'bcmlir', 'hivm'}
    suffix = 'mlir' if stage == 'hivm' else stage
    decision = read(call / 'before-edit-decision.json')
    assert decision['input_files'] == initial
    assert sha(workspace / 'diagnosis.json') == decision['diagnosis_sha256']
    assert read(workspace / 'diagnosis.json') == decision['diagnosis']
    uses, results = {}, {}
    for index, event in enumerate(read_events(call / 'agent-events.jsonl')):
        message = event.get('message', {})
        if not isinstance(message, dict):
            continue
        for block in message.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                uses[block['id']] = index, block
            elif block.get('type') == 'tool_result':
                results[block['tool_use_id']] = index, block
    bridges = []
    for key, (index, result) in results.items():
        if key not in uses:
            continue
        before, use = uses[key]
        if use['name'] == 'Bash' and str(use['input'].get('command', '')).strip().endswith(' decision'):
            if 'decision_recorded' in str(result.get('content')) and not result.get('is_error'):
                bridges.append((before, index))
    assert len(bridges) == 1
    decision_use, decision_result = bridges[0]
    edits = [index for index, use in uses.values() if use['name'] == 'Edit'
        and str(use['input'].get('file_path', '')).endswith('.' + suffix)]
    assert edits and min(edits) > decision_result
    evidence_reads = {}
    ir_tokens = ['.ttir', '.bcmlir'] if stage == 'ttir' else ['.' + suffix]
    evidence_tokens = ['context.json', 'profiling.json', 'hotspots.json', 'timeline.json', 'frozen.py', *ir_tokens]
    if read(workspace / 'context.json').get('evidence_mode') == 'compile-repair':
        evidence_tokens = ['context.json', 'feedback.json', 'frozen.py', '.' + suffix]
    for token in evidence_tokens:
        matches = []
        for key, (index, use) in uses.items():
            if index >= decision_use or use['name'] not in {'Read', 'Bash', 'Grep'} or token not in str(use['input']) or key not in results:
                continue
            result_index, result = results[key]
            if result_index < decision_use and not result.get('is_error') and result.get('content'):
                matches.append(dict(use_event=index, result_event=result_index, input=use['input']))
        assert matches, 'Missing successful pre-decision read: ' + token
        evidence_reads[token] = matches
    changes = {}
    diffs = []
    for entry in read(workspace / 'ir-inventory.json')['entries']:
        relative = entry['path']
        candidate = call / 'output-ir' / relative
        assert sha(candidate) == actual[relative]
        if initial[relative] == actual[relative]:
            continue
        original = call / 'input-ir' / relative
        a, b = original.read_text(), candidate.read_text()
        declaration = 'tt.func public' if stage == 'ttir' else 'func.func @'
        signatures = [x for x in a.splitlines() if declaration in x]
        assert signatures and signatures == [x for x in b.splitlines() if declaration in x], \
            'Public signature changed; explicit ABI review required'
        changes[relative] = sha(candidate)
        diffs.extend(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
            fromfile='input/' + relative, tofile='output/' + relative))
    assert changes and set(changes) <= editable
    (call / 'model-ir.diff').write_text(''.join(diffs), encoding='utf-8')
    record = dict(status='bindings_and_preedit_reads_verified', source_sha256=actual['frozen.py'],
        changed_ir=changes, decision_use_event=decision_use, decision_result_event=decision_result,
        first_ir_edit_event=min(edits), evidence_reads=evidence_reads, public_signature_identical=True,
        semantic_approval='Pending host review of actual diff and complete operation semantics')
    save(call / 'common-review.json', record)
    return record


def read_events(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(review(args.run.resolve(), args.invocation)))


if __name__ == '__main__':
    main()
