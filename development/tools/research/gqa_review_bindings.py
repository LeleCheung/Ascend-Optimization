"""Extract successful pre-edit evidence reads for a GQA model handoff."""
import argparse
import difflib
import json
from pathlib import Path
import re
import shlex

from ir_campaign import read, save, sha


def evidence_read(use, token, expected_text=None, output=None):
    if use['name'] == 'Read':
        return token in str(use['input'])
    if use['name'] != 'Bash':
        return False
    command = use['input'].get('command', '')
    if any(symbol in command for symbol in ['<', '>', '$', '`', '\n']):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        segments, arguments, separators = [], [], []
        for value in lexer:
            if value in {';', '&&', '|'}:
                if not arguments:
                    return False
                segments.append(arguments)
                separators.append(value)
                arguments = []
            elif value in {'&', '||'}:
                return False
            else:
                arguments.append(value)
        if not arguments:
            return False
        segments.append(arguments)
    except ValueError:
        return False
    found = False
    for index, arguments in enumerate(segments):
        name = arguments[0]
        if name not in {'echo', 'sha256sum', 'ls', 'cat', 'head', 'sed', 'grep'}:
            return False
        if name == 'sed':
            # Only printed line ranges, never sed programs that write or execute.
            if len(arguments) < 3 or arguments[1] != '-n' or not re.fullmatch(r'\d+(?:,\d+)?p', arguments[2]):
                return False
        if name in {'cat', 'head', 'sed', 'grep'} and len(arguments) >= 2:
            path = arguments[-1]
            if token in path and not path.startswith('-'):
                # The file-reading segment must succeed before chained successors.
                if index < len(separators) and separators[index] == ';':
                    if not expected_text or not output or expected_text.strip() not in output:
                        return False
                found = True
    return found


def review(run, invocation, ir_evidence=('.ttir', '.bcmlir'), paired_reader='model'):
    call = run / 'agent-calls' / invocation
    assert (call / 'model-usage.json').exists()
    assert sha(call / 'output-source.py') == sha(run / 'workspace/candidate.py')
    uses, results = {}, {}
    for index, line in enumerate((call / 'agent-events.jsonl').read_text().splitlines()):
        message = json.loads(line).get('message', {})
        if not isinstance(message, dict):
            continue
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                uses[block['id']] = index, block
            elif block.get('type') == 'tool_result':
                results[block['tool_use_id']] = index, block
    def recorded(result):
        if result.get('is_error'):
            return False
        try:
            value = json.loads(result['content'])
        except (TypeError, ValueError):
            return False
        if not isinstance(value, dict):
            return False
        return value.get('status') == 'recorded' and bool(value.get('decision_id')) and Path(value['path']).is_file()

    decisions = [(index, key) for key, (index, use) in uses.items()
        if use['name'] == 'Bash' and ' decision --record ' in use['input'].get('command', '')
        and key in results and recorded(results[key][1])]
    assert len(decisions) == 1
    decision_use, key = decisions[0]
    decision_result = results[key][0]
    edits = [index for index, use in uses.values() if use['name'] == 'Edit'
        and use['input'].get('file_path', '').endswith('/candidate.py')]
    assert edits and min(edits) > decision_result
    reads = {}
    config = read(call / 'config.json')
    if config.get('required_simulator_evidence'):
        tokens = (['current-profiling.json'] if paired_reader == 'model' else []) + [
            'simulator-evidence.json', *ir_evidence, 'candidate.py']
    elif config.get('evidence_mode') == 'profiling-source-ir':
        tokens = (['current-profiling.json'] if paired_reader == 'model' else []) + [
            *ir_evidence, 'candidate.py']
    else:
        tokens = ['repair-feedback.json', 'candidate.py']
    for token in tokens:
        expected = call / 'input-source.py' if token == 'candidate.py' else run / 'workspace' / token
        expected_text = expected.read_text() if expected.is_file() else None
        def expected_output(use):
            if expected_text or not token.startswith('.') or use['name'] != 'Bash':
                return expected_text
            command = use['input'].get('command', '')
            for start, end, filename in re.findall(
                    r'''sed\s+-n\s+['"]?(\d+)(?:,(\d+))?p['"]?\s+([^;\s]+)''', command):
                path = (run / 'workspace' / filename).resolve()
                if (path.is_relative_to((run / 'workspace').resolve()) and path.is_file()
                        and str(path).endswith(token)):
                    lines = path.read_text().splitlines()
                    return '\n'.join(lines[int(start)-1:int(end or start)])
            return None
        matches = [dict(use_event=index, result_event=results[key][0], tool=use['name'], input=use['input'])
            for key, (index, use) in uses.items() if index < decision_use
            and key in results and results[key][0] < decision_use
            and evidence_read(use, token, expected_output(use), str(results[key][1].get('content', '')))
            and not results[key][1].get('is_error')]
        assert matches, token
        reads[token] = matches
    a = (call / 'input-source.py').read_text()
    b = (call / 'output-source.py').read_text()
    (call / 'source.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True))))
    record = dict(status='preedit_reads_verified_semantics_pending',
        source_sha256=sha(call / 'output-source.py'), parent_source_sha256=sha(call / 'input-source.py'),
        decision_use_event=decision_use, decision_result_event=decision_result,
        first_source_edit=min(edits), evidence_reads=reads)
    record['required_ir_evidence'] = list(ir_evidence)
    record['paired_profiling_reader'] = paired_reader
    record['evidence_mode'] = config.get('evidence_mode', 'historical_default')
    save(call / 'common-review.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--ir-evidence', action='append', choices=['.ttir', '.bcmlir'],
                        help='Explicit IR level required for this source-level mechanism')
    args = parser.parse_args()
    print(review(args.run.resolve(), args.invocation, args.ir_evidence or ('.ttir', '.bcmlir')))
