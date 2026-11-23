"""Verify full-contract binary changes against the registered IR edit set."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def formal_receipts(evaluation):
    """Separate formal compiler receipts from later supplementary processes."""
    directory = evaluation / 'compiler-receipts'
    snapshot = evaluation / 'formal-compiler-receipts.json'
    if snapshot.exists():
        scope = read(snapshot)
        paths = []
        for row in scope['files']:
            path = (evaluation / row['path']).resolve()
            assert path.is_relative_to(directory.resolve())
            assert sha(path) == row['sha256'], 'Formal compiler receipt bytes changed'
            paths.append(path)
        assert paths
        return paths, dict(method='recorded_before_supplements', snapshot_sha256=sha(snapshot),
                           files=scope['files'])
    # Older evaluations saved result.json before running supplementary checks.
    # Preserve that explicit reconstruction method instead of mixing their ASTs.
    cutoff = (evaluation / 'result.json').stat().st_mtime_ns
    all_paths = sorted(directory.glob('*.jsonl'))
    paths = [path for path in all_paths if path.stat().st_mtime_ns <= cutoff]
    assert paths, 'No closed compiler receipts precede the saved formal result'
    expected = read(evaluation / 'compiler-verification.json')['receipts']
    assert sum(len([line for line in path.read_text().splitlines() if line.strip()])
               for path in paths) == expected, 'Reconstructed receipt count differs from formal verification'
    return paths, dict(method='reconstructed_by_closed_file_time_before_formal_result',
         result_mtime_ns=cutoff,
         files=[dict(path=path.relative_to(evaluation).as_posix(), sha256=sha(path)) for path in paths],
         excluded_later_files=[path.relative_to(evaluation).as_posix() for path in all_paths if path not in paths])


def binaries(evaluation, source, receipts=None):
    mapped = {}
    if receipts is None:
        receipts = sorted((evaluation / 'compiler-receipts').glob('*.jsonl'))
    assert receipts, 'Missing compiler receipts'
    for path in receipts:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row['source_sha256'] == source
            binary = next(item for item in row['artifacts'] if item['kind'] == 'npubin')
            assert sha(binary['path']) == binary['sha256'], 'Compiler binary bytes changed'
            key = row['ast_sha256']
            assert key not in mapped or mapped[key] == binary['sha256'], 'AST has multiple binary identities'
            mapped[key] = binary['sha256']
    assert mapped
    return mapped


def compare(run, invocation):
    original = run / 'evaluations/unchanged-ir'
    candidate = run / 'evaluations' / invocation
    config = read(run / 'config.json')
    source = config['source_sha256']
    original_files, original_scope = formal_receipts(original)
    candidate_files, candidate_scope = formal_receipts(candidate)
    a = binaries(original, source, original_files)
    b = binaries(candidate, source, candidate_files)
    assert set(a) == set(b), 'Formal specialization sets differ'
    manifest = read(candidate / 'implementation/ir-manifest.json')
    registered = {row['ast_sha256'] for row in manifest['overrides']
                  if row['sha256'] != row['original_sha256']}
    changed = {key for key in a if a[key] != b[key]}
    assert registered and changed <= registered, 'An unedited specialization changed binary'
    assert registered <= set(a), 'Registered edit absent from the full contract'
    results = [read(folder / 'result.json') for folder in [original, candidate]]
    assert all(row['status'] == 'PASSED' and row['num_passed'] == row['num_workloads'] for row in results)
    cases = [{row['uuid']: row for row in result['per_workload'] if row['phase'] == 'timing'}
             for result in results]
    assert set(cases[0]) == set(cases[1])
    timings = []
    for name in sorted(cases[0]):
        old, new = [case[name] for case in cases]
        assert old['status'] == new['status'] == 'PASSED'
        assert all(math.isfinite(row[key]) and row[key] > 0 for row in [old, new]
                   for key in ['latency_ms', 'reference_latency_ms'])
        timings.append(dict(case=name, original_us=old['latency_ms'] * 1000,
            candidate_us=new['latency_ms'] * 1000,
            original_over_candidate=old['latency_ms'] / new['latency_ms'],
            original_reference_us=old['reference_latency_ms'] * 1000,
            candidate_reference_us=new['reference_latency_ms'] * 1000))
    return dict(status='exact_untouched_binaries_verified', source_sha256=source,
        invocation=invocation, ast_count=len(a), changed_ast_count=len(changed),
        unchanged_ast_count=len(a) - len(changed), registered_edited_asts=sorted(registered),
        changed=[dict(ast_sha256=key, original_binary_sha256=a[key], candidate_binary_sha256=b[key])
                 for key in sorted(changed)],
        all_binary_identities=[dict(ast_sha256=key, original_binary_sha256=a[key], candidate_binary_sha256=b[key])
                               for key in sorted(a)],
        full_contract_candidate_latency_geomean_ratio=math.exp(sum(math.log(row['original_over_candidate'])
            for row in timings) / len(timings)), timing_cases=timings,
        formal_result_sha256=[sha(folder / 'result.json') for folder in [original, candidate]],
        formal_receipt_scopes=dict(original=original_scope, candidate=candidate_scope),
        comparison_tool_sha256=sha(__file__),
        scope='Exact full-contract binary identity comparison. Same-binary timing changes and per-case reference drift remain separate measurement observations.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    result = compare(args.run.resolve(), args.invocation)
    output = args.run / (args.invocation.replace('-', '') + '-binary-comparison.json')
    with output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'ast_count', 'changed_ast_count',
        'unchanged_ast_count', 'full_contract_candidate_latency_geomean_ratio']}))


if __name__ == '__main__':
    main()
