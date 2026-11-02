"""在原卡锁边界采集候选/原参考的宿主与设备诊断，不重复正式评测。"""
from pathlib import Path


def pending(previous, outputs):
    missing = [path for path in outputs if not path.exists()]
    if missing and previous.get('status') in {'running', 'completed'}:
        raise RuntimeError('组件诊断有未知终态，不重发：' + str(missing[0]))
    return missing


def process(runner):
    from night_common import HERE, PY, read, save, sha
    folder = runner.batch / 'component-diagnostics/expm1-components-v1'
    spec_file = folder / 'spec.json'
    if not spec_file.exists():
        return
    spec = read(spec_file)
    assert spec['qualified_CANN'] is False and spec['formal_performance_measured'] is False
    checker = HERE / 'component_probe.py'
    assert spec['checker_sha256'] == sha(checker)
    key = 'diagnostic:expm1-components-v1'
    outputs = [folder / item['name'] / 'result.json' for item in spec['items']]
    previous = runner.state['jobs'].get(key, {})
    pending(previous, outputs)
    result_file = folder / 'result.json'
    if result_file.exists():
        value = read(result_file)
        assert value['spec_sha256'] == sha(spec_file)
        for name, digest in value['evidence_sha256'].items():
            assert sha(Path(name)) == digest
        return
    frozen = folder / ('checker-' + sha(checker) + '.py')
    if frozen.exists():
        assert frozen.read_bytes() == checker.read_bytes()
    else:
        frozen.write_bytes(checker.read_bytes())
    current = runner.state.get('current_job')
    runner.state['current_job'] = key
    runner.state['jobs'][key] = dict(status='running', spec_sha256=sha(spec_file))
    runner.persist()
    rows = []
    evidence = {str(spec_file): sha(spec_file), str(frozen): sha(frozen)}
    for item, output in zip(spec['items'], outputs):
        payload = item['spec']
        source = Path(payload['source'])
        assert payload['source_sha256'] == sha(source)
        assert spec['oracle_sha256'] == sha(Path(payload['leaf']) / 'oracle.py')
        case_file = Path(payload['leaf']) / 'timing.jsonl'
        assert spec['timing_sha256'] == sha(case_file)
        cases = [__import__('json').loads(line) for line in case_file.read_text().splitlines() if line.strip()]
        assert payload['case'] in cases, '只能使用冻结原输入'
        task_spec = folder / (item['name'] + '-spec.json')
        if task_spec.exists():
            assert read(task_spec) == payload
        else:
            save(task_spec, payload)
        if not output.exists():
            assert not output.parent.exists(), '保留未完成原件，不覆盖采集'
            runner.cmd(key + ':' + item['name'],
                       [PY, frozen, '--spec', task_spec, '--output', output.parent], 1200)
        value = read(output)
        assert value['status'] == 'completed' and value['spec'] == payload
        assert value['probe_sha256'] == sha(frozen)
        trace = Path(value['device_trace'])
        assert sha(trace) == value['device_trace_sha256']
        evidence.update({str(source): sha(source), str(task_spec): sha(task_spec),
                         str(output): sha(output), str(trace): sha(trace)})
        rows.append(dict(name=item['name'], side=item['side'], source_sha256=sha(source),
            case=payload['case']['name'], host_call_us=value['host_call_us'],
            synchronized_call_us=value['synchronized_call_us'],
            device_task_union_us=value['device_task_union_us'],
            tasks=[dict(name=task['name'], duration_us=task['dur'],
                        task_type=task.get('args', {}).get('Task Type')) for task in value['tasks']]))
    result = dict(status='measured', spec_sha256=sha(spec_file), rows=rows,
        qualified_CANN=False, formal_performance_measured=False,
        scope='原PyTorch只作诊断对照；宿主样本与带profiler设备时间不相减、不相加，不作为新增正式测量或CANN资格。',
        evidence_sha256=evidence)
    save(result_file, result)
    runner.state['jobs'][key].update(status='completed', result=str(result_file),
                                   result_sha256=sha(result_file), outcome='measured')
    runner.state['current_job'] = current
    runner.persist()
