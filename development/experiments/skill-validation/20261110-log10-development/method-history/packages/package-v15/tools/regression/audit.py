"""离线核验独立回归的源码、合同、请求、逐 case 结果及两侧补测。"""
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


def relocate_job(job, original_run, local_run):
    """只在内存映射归档位置，保留原始请求、合同及身份文件的字节。"""
    root = PurePosixPath(original_run)
    assert root.is_absolute() and '..' not in root.parts

    def local(value):
        path = PurePosixPath(value)
        assert path.is_absolute() and '..' not in path.parts, '原件路径不规范'
        relative = path.relative_to(root)
        return str(local_run.joinpath(*relative.parts))

    mapped = copy.deepcopy(job)
    for name in ['source', 'review', 'archive', 'request']:
        mapped[name] = local(mapped[name])
    for spec in mapped['checks'].values():
        for name in ['leaf', 'checker']:
            spec[name] = local(spec[name])
    return mapped


def inspect_transfer_manifest(run):
    """核验传输清单的全部原件，包含不直接参与分数重算的二进制与编译产物。"""
    path = run/'transfer-manifest.json'
    manifest = read(path if path.exists() else run/'pull-manifest.json')
    names = set()
    for item in manifest:
        relative = PurePosixPath(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.as_posix() not in names, '传输清单含重复路径'
        names.add(relative.as_posix())
        path = run.joinpath(*relative.parts)
        assert path.resolve().is_relative_to(run.resolve())
        assert path.stat().st_size == item['bytes'], '传输原件长度改变：'+item['path']
        assert sha(path) == item['sha256'], '传输原件身份改变：'+item['path']
    return len(names)


def inspect_formal(result, expected_cases):
    """从完整 case 记录重算计数、计时比值和几何平均。"""
    cases = result['per_workload']
    assert Counter((case['uuid'], case['phase']) for case in cases) == Counter(expected_cases), '正式 case 缺失或多出'
    assert result['num_workloads'] == len(expected_cases)
    assert result['num_passed'] == sum(case['status'] == 'PASSED' for case in cases)
    assert result['server_backend'] == 'npu', '实际后端发生变化'
    timing = [case for case in cases if case['phase'] == 'timing']
    positive = all(all(isinstance(case.get(key), (float,int)) and math.isfinite(case[key]) and case[key] > 0
                       for key in ['speedup','latency_ms','reference_latency_ms']) for case in timing)
    if positive and timing:
        for case in timing:
            assert math.isclose(case['speedup'],case['reference_latency_ms']/case['latency_ms'],rel_tol=1e-9)
        geo_mean = math.exp(sum(math.log(case['speedup']) for case in timing)/len(timing))
        assert math.isclose(result['geo_mean'],geo_mean,rel_tol=1e-9), '几何平均无法由逐 case 重建'
    else:
        geo_mean = None
    full = (result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
            and result.get('is_hack') is False and result.get('timing_skipped') is False
            and positive and bool(timing))
    return dict(full_passed=bool(full), geo_mean=geo_mean,formal_cases=len(cases),timing_cases=len(timing))


def inspect_job(job, run):
    """证据完整性与验收结论分开：性能未过线不是证据损坏。"""
    root = run/'operators'/job['operator']
    assert sha(job['source']) == job['source_sha256'], '候选字节身份改变'
    source = Path(job['source']).read_bytes().decode('utf8')
    review = read(job['review'])
    admit(source, review)
    gate = read(root/'gate.json')['result']
    assert gate['source_sha256'] == job['source_sha256'] and gate['admission'] == 'review_bound'
    assert 'sha256:'+sha(job['archive']) == job['bundle_id'], '合同包身份改变'
    request = read(job['request'])
    submitted = read(root/'evaluation/request.json')
    assert submitted == request, '正式请求不等于准备计划'
    assert submitted['implementation']['sources'][0]['content'] == source
    assert submitted['binding']['bundle_id'] == job['bundle_id']
    assert submitted['settings'] == job['settings']
    operation = read(root/'evaluation/operation.json')
    assert operation['source_sha256'] == job['source_sha256']
    assert operation['request']['sha256'] == sha(root/'evaluation/request.json')
    checks = {}
    expected_cases = []
    for kind, spec in job['checks'].items():
        assert sha(spec['checker']) == spec['checker_sha256'], '检查器身份改变'
        for name, digest in spec['contract_sha256'].items():
            assert sha(Path(spec['leaf'])/name) == digest, '合同文件身份改变：'+name
        value = read(root/(kind+'.json'))
        assert value['source_sha256'] == job['source_sha256']
        assert value['checker_source']['sha256'] == spec['checker_sha256']
        assert value['total'] == spec['previous_total'], '固定补测数量发生变化'
        assert len(value['cases']) == value['total'], '补测逐项记录不齐全'
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
    parser.add_argument('--original-run-directory',help='异机归档中的原始Linux运行目录；只在内存映射路径')
    parser.add_argument('--verify-transfer',action='store_true',help='同时核验transfer/pull-manifest.json中的全部原件')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run_directory.resolve()
    plan = read(run/'plan.json')
    state = read(run/'state.json')
    transfer_files = inspect_transfer_manifest(run) if args.verify_transfer else None
    assert not plan['errors'] and len(plan['jobs']) == plan['expected']
    assert state['status'] == 'completed', '队列仍在运行，不能生成最终核验'
    rows = []
    errors = []
    for job in plan['jobs']:
        try:
            mapped = relocate_job(job,args.original_run_directory,run) if args.original_run_directory else job
            row = inspect_job(mapped,run)
            assert state['operators'][job['operator']]['accepted'] == row['accepted'], '控制器验收与原件重算不一致'
            rows.append(row)
        except Exception as error:
            errors.append(dict(operator=job['operator'],error=type(error).__name__+': '+str(error)))
    value = dict(schema_version='hgkda.regression-audit/v1',
                 evidence_verified=not errors,expected=plan['expected'],verified=len(rows),
                 accepted=sum(row['accepted'] for row in rows),rows=rows,errors=errors,
                 transfer_files_verified=transfer_files,
                 scope='离线核验本次真实新入口回归；不提交设备任务，不替代运行时后端审查')
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value[key] for key in ['evidence_verified','expected','verified','accepted','errors']}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
