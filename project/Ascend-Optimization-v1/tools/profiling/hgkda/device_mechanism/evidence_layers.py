"""采集后的统一证据导出：真机、IR、模拟器；RTL只登记接口，不参与本批选版。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from audit_artifacts import Evidence

LAYERS = {'silicon', 'compiler_ir', 'simulator', 'rtl'}
STATUSES = {'available', 'partial', 'failed', 'not_run'}


def validate(record):
    if record['layer'] not in LAYERS or record['status'] not in STATUSES:
        raise ValueError('证据层级或状态不合法')
    digest = record['source_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('源码身份必须为SHA256')
    if record['status'] == 'available' and not record['artifacts']:
        raise ValueError('可用证据必须有原始附件')
    if record['layer'] == 'rtl' and record['status'] != 'not_run':
        raise ValueError('当前没有RTL执行器，不能登记RTL运行结果')
    for metric in record['measurements']:
        if not metric.get('unit') or not metric.get('scope'):
            raise ValueError('测量必须写清单位与范围')


def export(batch, archive=None):
    evidence = Evidence(batch, archive)
    plan = evidence.json('plan.json'); records = []
    for entry in plan['records']:
        op = entry['operator']; root = PurePosixPath('results') / op / 'development'
        probe = evidence.json(root / 'interventions/probe.json')
        for case in probe['cases']:
            base = dict(operator=op, source_sha256=probe['source_sha256'],
                        input=dict(spec=case['spec'], generator_seed=71439 + int(case['id']), origin='device_intervention'),
                        case_id=case['id'], physical_device=7)
            for name, launches, digest in [('parent', case['parent_launches'], probe['source_sha256'])] + [
                    (v['name'], v.get('launches', []), v['source_sha256']) for v in case['variants'] if v['status'] == 'passed']:
                records.append(dict(base, layer='compiler_ir', status='available', variant=name, source_sha256=digest,
                    artifacts=[dict(path=evidence.name(a['path']),sha256=a['sha256'],kind=a['kind']) for launch in launches for a in launch['artifacts']],
                    measurements=[], limits=['IR为静态编译表示，不代表动态指令计数或完整依赖图']))
            for variant in case['variants']:
                if variant['status'] != 'passed': continue
                artifacts = []
                for repeat in range(3):
                    name = root / 'interventions' / ('case-' + case['id']) / variant['name'] / ('benchmark-' + str(repeat)) / 'measurement.json'
                    measurement = evidence.json(name)
                    artifacts.append(dict(path=evidence.name(measurement['csv_path']),sha256=measurement['csv_sha256'],kind='task_time_csv'))
                records.append(dict(base, layer='silicon', status='available', variant=variant['name'],
                    source_sha256=variant['source_sha256'], reference_source_sha256=probe['source_sha256'], artifacts=artifacts,
                    measurements=[dict(name=key, values=values, unit='ms', scope='captured_single_kernel_device_task') for key, values in variant['timings'].items()],
                    limits=['固定指针设备提交，排除host发射间隙；不代表完整调用延迟']))
        contract = evidence.name(entry['splits']['development']['path'] + '/timing.jsonl')
        cases = {row['name']:row for row in [json.loads(line) for line in evidence.read(contract).decode().splitlines() if line]}
        for name in evidence.names('/facts.json'):
            if not name.startswith(str(root) + '/') or not any(token in PurePosixPath(name).parent.name for token in ['metrics', 'instruction']):
                continue
            facts = evidence.json(name); directory = PurePosixPath(name).parent
            identity = evidence.json(directory / 'identity.json')
            artifacts = evidence.json(directory / 'artifacts.json')
            instruction = identity['level'] == 'instruction'
            complete = bool(facts.get('simulator_hotspots')) if instruction else bool((facts.get('metrics') or {}).get('ops'))
            records.append(dict(operator=op, source_sha256=identity['source_sha256'], physical_device=7,
                case_id=identity['case_id'], input=dict(contract=contract, case=cases[identity['case_id']], origin='kgs_profile'),
                layer='simulator' if instruction else 'silicon',
                status='available' if facts['profile_status'] == 'completed' and complete else 'partial' if facts.get('metrics') else 'failed',
                artifacts=[dict(path=evidence.name(a['path']),sha256=a['sha256'],kind=a['kind']) for a in artifacts],
                measurements=[], fact_file=name, acquisition_budget_seconds=identity['budget_seconds'],
                elapsed_seconds=facts['seconds'], service_status=facts['profile_status'],
                limits=['指标单位和范围保留于原始facts；模拟器周期不可替代真机时间']))
        records.append(dict(operator=op, source_sha256=entry['source_sha256'], physical_device=None, case_id=None, input=None,
            layer='rtl',status='not_run',artifacts=[],measurements=[],
            required_adapter_fields=['design_revision','clock_period_ns','memory_model','stimulus_sha256','waveform_sha256','cycle_scope'],
            limits=['接口预留；没有RTL设计、仿真器或运行结果，不据此作机制结论']))
    for record in records: validate(record)
    if evidence.tar: evidence.tar.close()
    return dict(schema='hgkda-layered-evidence-v1',batch=plan['batch'],records=records,
        provenance='采集后只读导出；未增加冻结模型输入、未改选版；RTL仅接口',
        cross_layer_policy='先按源码、输入、编译产物和条件绑定，再比较各自可观测量；不跨层相加或替代时间')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True);parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise ValueError('不覆盖证据导出')
    value=export(args.batch_dir,args.archive)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(schema=value['schema'],records=len(value['records'])),ensure_ascii=False))
