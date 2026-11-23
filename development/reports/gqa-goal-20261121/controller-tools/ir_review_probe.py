"""Review model-authored third-probe scalarization without editing candidate IR."""
import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re

from ir_campaign import read, save, sha


def evidence_review(call):
    uses, results = {}, {}
    for index, line in enumerate((call / 'agent-events.jsonl').read_text().splitlines()):
        message = json.loads(line).get('message', {})
        if not isinstance(message, dict):
            continue
        for block in message.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                uses[block['id']] = index, block
            elif block.get('type') == 'tool_result':
                results[block['tool_use_id']] = index, block
    decisions = [(uses[key][0], index) for key, (index, result) in results.items()
        if key in uses and uses[key][1]['name'] == 'Bash'
        and str(uses[key][1]['input'].get('command', '')).strip().endswith(' decision')
        and 'decision_recorded' in str(result.get('content')) and not result.get('is_error')]
    assert len(decisions) == 1
    decision_use, decision_result = decisions[0]
    edits = [index for index, use in uses.values() if use['name'] == 'Edit'
        and str(use['input'].get('file_path', '')).endswith('.ttir')]
    assert edits and min(edits) > decision_result
    reads = {}
    for token in ['context.json', 'feedback.json', 'validation-policy.json',
            'profiling.json', 'hotspots.json', 'timeline.json', 'frozen.py',
            '.ttir', '.ttadapter', '.bcmlir']:
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
        first_ir_edit_event_index=min(edits), evidence_reads=reads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    submission = read(call / 'submission.json')
    assert submission['status'] == 'awaiting_ir_review'
    assert (call / 'model-usage.json').exists()
    inventory = read(run / 'workspace/ir-inventory.json')
    changed = []
    for entry in inventory['entries']:
        original = call / 'input-ir' / entry['path']
        candidate = call / 'output-ir' / entry['path']
        if original.read_bytes() != candidate.read_bytes():
            changed.append((entry, original, candidate))
    assert len(changed) == 1
    entry, original, candidate = changed[0]
    assert entry['path'].endswith('/_scatter_add_sort_out_kernel.ttir')
    a, b = original.read_text(), candidate.read_text()
    assert a.count('tt.gather') == 11 and b.count('tt.gather') == 10
    old_gather = next(line for line in a.splitlines(True) if '%go_24 = tt.gather' in line)
    extra_results = {'%probe_low', '%probe_high', '%probe_3'}
    for offset in [256, 768, 1280, 1792]:
        extra_results.update({f'%c{offset}_i32', f'%key_{offset}_p',
            f'%key_{offset}', f'%key_{offset}_v'})
        assert re.search(rf'%c{offset}_i32 = arith.constant {offset} : i32', b)
        assert f'%key_{offset}_p = tt.addptr %keys, %c{offset}_i32 : !tt.ptr<f32>, i32' in b
        assert f'%key_{offset} = tt.load %key_{offset}_p : !tt.ptr<f32>' in b
        assert f'%key_{offset}_v = tt.splat %key_{offset} : f32 -> tensor<2048xf32>' in b
    assert '%probe_low = arith.select %go_17, %key_768_v, %key_256_v' in b
    assert '%probe_high = arith.select %go_17, %key_1792_v, %key_1280_v' in b
    assert '%probe_3 = arith.select %go_11, %probe_high, %probe_low' in b
    normalized = []
    for line in b.splitlines(True):
        if line.strip().split(' = ', 1)[0] in extra_results:
            continue
        if '%go_25 = arith.cmpf' in line:
            normalized.append(old_gather)
            line = line.replace('%probe_3,', '%go_24,')
        normalized.append(line)
    assert ''.join(normalized) == a, 'Changes extend beyond the reviewed probe substitution'
    # These four cases cover arbitrary outcomes of the first two comparisons,
    # including outcomes that do not arise from a sorted finite key vector.
    supports = []
    for first in [False, True]:
        for second in [False, True]:
            lo, hi = (1025, 2048) if first else (0, 1024)
            mid = min((lo + hi) >> 1, 2047)
            lo, hi = (mid + 1, hi) if second else (lo, mid)
            third = min((lo + hi) >> 1, 2047)
            selected = (1792 if second else 1280) if first else (768 if second else 256)
            assert third == selected and 0 <= selected < 2048
            supports.append(dict(first=first, second=second, original_index=third, selected_index=selected))
    # Reuse the existing abstract lower-bound domain, without adding device cases.
    for rank in range(2049):
        lo, hi = 0, 2048
        for step in range(11):
            mid = min((lo + hi) >> 1, 2047)
            if step == 0:
                first = mid < rank
            elif step == 1:
                second = mid < rank
            elif step == 2:
                selected = (1792 if second else 1280) if first else (768 if second else 256)
                assert mid == selected
            lo, hi = (mid + 1, hi) if mid < rank else (lo, mid)
        lo += int(lo == 0 and 0 < rank)
        assert lo == rank
    evidence_review(call)
    proof = dict(status='passed', kind='third_probe_address_identity',
        boolean_combinations=supports, abstract_insertion_ranks=2049,
        arithmetic_and_remaining_ir_identical=True,
        scope='Exact same key bits selected; existing tail fold and all subsequent arithmetic unchanged')
    save(call / 'semantic-proof.json', proof)
    (call / 'reviewed.diff').write_text(''.join(difflib.unified_diff(
        a.splitlines(True), b.splitlines(True), fromfile='input.ttir', tofile='output.ttir')),
        encoding='utf8')
    save(call / 'ir-review.json', dict(decision='approved',
        reviewed_at=datetime.now(timezone.utc).isoformat(), reviewer='Codex',
        source_sha256=inventory['source_sha256'], changed_ir={entry['path']: sha(candidate)},
        decision_sha256=sha(call / 'before-edit-decision.json'), semantic_proof=proof,
        checks=['Only third probe substitution; source, ABI, tail fold, sum and output unchanged',
            'Four new reads address immutable row-local keys within the existing2048-element read',
            'Pre-edit profiling, simulator hotspots, timeline and actual lowerings read successfully'],
        claim_corrections=['Fixed small support exposes a profitable opportunity only after device measurement',
            'MTE2 counters alone do not establish scalar-load cost; added scalar operations remain measurable',
            'Gather decrease is established at TTIR; exact BAR/wait changes require compiled replay',
            'Four new scalar loads add16 logical GM bytes per program before backend/cache effects',
            'Partial same-core windows retain their own coverage; task-level latency comes from real hardware'],
        layer_choice_scope='Fixed-source IR isolation; equivalent scalar probes can also be expressed in Triton',
        validation_required=['CPU lowering', 'fixed complete KGS and existing supplements',
            'post profiling and isolated real-input simulator replay', 'ABBA only for concrete timing concern'],
        scope='Approved for CPU lowering and fixed device checks; no performance-retention conclusion'))
    print(json.dumps(dict(status='host_review_passed', ranks=2049, gather_before=11, gather_after=10)))


if __name__ == '__main__':
    main()
