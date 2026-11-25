"""Audit regression source, contracts, requests, per-case results, and both supplementary checks offline."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from hgkda.gate import admit
from hgkda.validation import supplement_status


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relocate_job(job, original_run, local_run, dependency_roots=None):
    """Map archive paths in memory while preserving raw request, contract, and identity bytes."""
    root = PurePosixPath(original_run)
    assert root.is_absolute() and '..' not in root.parts
    roots=[(root,local_run)]
    for original,target in (dependency_roots or {}).items():
        prefix=PurePosixPath(original)
        assert prefix.is_absolute() and '..' not in prefix.parts
        roots.append((prefix,Path(target)))
    roots.sort(key=lambda pair:len(pair[0].parts),reverse=True)

    def local(value):
        path = PurePosixPath(value)
        assert path.is_absolute() and '..' not in path.parts, 'Non-canonical artifact path'
        for prefix,target in roots:
            if path.is_relative_to(prefix):return str(target.joinpath(*path.relative_to(prefix).parts))
        raise ValueError('Undeclared archive dependency: '+value)

    mapped = copy.deepcopy(job)
    for name in ['source', 'review', 'archive', 'request']:
        mapped[name] = local(mapped[name])
    for spec in mapped['checks'].values():
        for name in ['leaf', 'checker']:
            spec[name] = local(spec[name])
    return mapped


def inspect_transfer_manifest(run):
    """Verify every transfer artifact, including binaries and compiler outputs not used in score reconstruction."""
    path = run/'transfer-manifest.json'
    if not path.exists():path=run/'pull-manifest.json'
    manifest=read(path if path.exists() else run/'export-manifest.json')
    if isinstance(manifest,dict):manifest=manifest['files']
    names = set()
    for item in manifest:
        relative = PurePosixPath(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.as_posix() not in names, 'Duplicate paths in transfer manifest'
        names.add(relative.as_posix())
        path = run.joinpath(*relative.parts)
        assert path.resolve().is_relative_to(run.resolve())
        assert path.stat().st_size == item['bytes'], 'Transferred artifact length changed: '+item['path']
        assert sha(path) == item['sha256'], 'Transferred artifact identity changed: '+item['path']
    return len(names)


def inspect_formal(result, expected_cases):
    """Reconstruct case counts, timing ratios, and geometric mean from full records."""
    cases = result['per_workload']
    assert Counter((case['uuid'], case['phase']) for case in cases) == Counter(expected_cases), 'Formal cases are missing or unexpected'
    assert result['num_workloads'] == len(expected_cases)
    assert result['num_passed'] == sum(case['status'] == 'PASSED' for case in cases)
    assert result['server_backend'] == 'npu', 'Actual backend changed'
    timing = [case for case in cases if case['phase'] == 'timing']
    positive = all(all(isinstance(case.get(key), (float,int)) and math.isfinite(case[key]) and case[key] > 0
                       for key in ['speedup','latency_ms','reference_latency_ms']) for case in timing)
    if positive and timing:
        for case in timing:
            assert math.isclose(case['speedup'],case['reference_latency_ms']/case['latency_ms'],rel_tol=1e-9)
        geo_mean = math.exp(sum(math.log(case['speedup']) for case in timing)/len(timing))
        assert math.isclose(result['geo_mean'],geo_mean,rel_tol=1e-9), 'Cannot reconstruct geometric mean from per-case records'
    else:
        geo_mean = None
    full = (result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
            and result.get('is_hack') is False and result.get('timing_skipped') is False
            and positive and bool(timing))
    return dict(full_passed=bool(full), geo_mean=geo_mean,formal_cases=len(cases),timing_cases=len(timing))


def inspect_job(job, run):
    """Separate evidence integrity from acceptance: a below-threshold score is not corrupted evidence."""
    root = run/'operators'/job['operator']
    assert sha(job['source']) == job['source_sha256'], 'Candidate byte identity changed'
    source = Path(job['source']).read_bytes().decode('utf8')
    review = read(job['review'])
    admit(source, review)
    gate = read(root/'gate.json')['result']
    assert gate['source_sha256'] == job['source_sha256'] and gate['admission'] == 'review_bound'
    assert 'sha256:'+sha(job['archive']) == job['bundle_id'], 'Contract archive identity changed'
    request = read(job['request'])
    submitted = read(root/'evaluation/request.json')
    assert submitted == request, 'Formal request differs from preparation plan'
    assert submitted['implementation']['sources'][0]['content'] == source
    assert submitted['binding']['bundle_id'] == job['bundle_id']
    assert submitted['settings'] == job['settings']
    operation = read(root/'evaluation/operation.json')
    assert operation['source_sha256'] == job['source_sha256']
    assert operation['request']['sha256'] == sha(root/'evaluation/request.json')
    checks = {}
    expected_cases = []
    for kind, spec in job['checks'].items():
        assert sha(spec['checker']) == spec['checker_sha256'], 'Checker identity changed'
        for name, digest in spec['contract_sha256'].items():
            assert sha(Path(spec['leaf'])/name) == digest, 'Contract file identity changed: '+name
        value = read(root/(kind+'.json'))
        assert value['source_sha256'] == job['source_sha256']
        assert value['checker_source']['sha256'] == spec['checker_sha256']
        assert value['total'] == spec['previous_total'], 'Fixed supplementary-check count changed'
        assert len(value['cases']) == value['total'], 'Supplementary per-case records are incomplete'
        assert sum(bool(case['passed']) for case in value['cases']) == value['passed']
        checks[kind] = dict(status=supplement_status(job['operator'], value, job['source_sha256']),
                            passed=value['passed'], total=value['total'])
        if kind == 'cann_supplementary':
            for phase in ['correctness', 'timing']:
                for line in (Path(spec['leaf'])/(phase+'.jsonl')).read_text().splitlines():
                    if line.strip():
                        case = json.loads(line)
                        expected_cases.append((case['name'], phase))
    result = read(root/'evaluation/result.json')
    formal = inspect_formal(result,expected_cases)
    accepted = bool(formal['full_passed'] and all(value['status'] == 'passed' for value in checks.values())
                    and formal['geo_mean'] > .8)
    return dict(operator=job['operator'],source_sha256=job['source_sha256'],
                bundle_id=job['bundle_id'],operation_id=operation['operation_id'],
                evidence_verified=True,accepted=accepted,**formal,checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--original-run-directory',help='Original Linux run directory in a cross-host archive; map paths in memory only')
    parser.add_argument('--verify-transfer',action='store_true',help='Also verify all artifacts in transfer/pull-manifest.json')
    parser.add_argument('--dependency-map',type=Path,help='Explicit JSON mapping from original Linux dependency roots to local archive roots; preserve originals')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run_directory.resolve()
    plan = read(run/'plan.json')
    state = read(run/'state.json')
    transfer_files = inspect_transfer_manifest(run) if args.verify_transfer else None
    dependencies={key:(args.dependency_map.parent/value).resolve() for key,value in read(args.dependency_map).items()} if args.dependency_map else None
    assert not plan['errors'] and len(plan['jobs']) == plan['expected']
    assert state['status'] == 'completed', 'Queue is still running; final verification cannot be generated'
    rows = []
    errors = []
    for job in plan['jobs']:
        try:
            mapped = relocate_job(job,args.original_run_directory,run,dependencies) if args.original_run_directory else job
            row = inspect_job(mapped,run)
            assert state['operators'][job['operator']]['accepted'] == row['accepted'], 'Controller acceptance differs from raw-artifact reconstruction'
            rows.append(row)
        except Exception as error:
            errors.append(dict(operator=job['operator'],error=type(error).__name__+': '+str(error)))
    value = dict(schema_version='hgkda.regression-audit/v1',
                 evidence_verified=not errors,expected=plan['expected'],verified=len(rows),
                 accepted=sum(row['accepted'] for row in rows),rows=rows,errors=errors,
                 transfer_files_verified=transfer_files,
                 scope='Offline audit of an actual fresh-entry regression; no device submission or substitute for runtime backend review')
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value[key] for key in ['evidence_verified','expected','verified','accepted','errors']}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
