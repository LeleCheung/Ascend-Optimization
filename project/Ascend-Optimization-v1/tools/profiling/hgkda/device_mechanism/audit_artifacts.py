"""独立核验设备计时CSV、编译IR和KGS附件；只读证据，不参与诊断或选版。"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile


class Evidence:
    def __init__(self, batch, archive):
        self.batch = batch
        self.tar = tarfile.open(archive, 'r:gz') if archive else None
        self.members = {m.name: m for m in self.tar.getmembers() if m.isfile()} if self.tar else {}

    def name(self, path):
        value = str(path).replace('\\', '/')
        marker = '/operators/campaigns/' + self.batch.name + '/'
        if marker in value:
            value = value.split(marker, 1)[1]
        name = PurePosixPath(value)
        if name.is_absolute() or '..' in name.parts or ':' in value:
            raise ValueError('附件不属于本批次：' + value)
        return name.as_posix()

    def read(self, path):
        name = self.name(path)
        if name in self.members:
            return self.tar.extractfile(self.members[name]).read()
        return (self.batch / name).read_bytes()

    def json(self, path):
        return json.loads(self.read(path).decode('utf-8-sig'))

    def names(self, suffix):
        names = {p.relative_to(self.batch).as_posix() for p in self.batch.rglob('*') if p.is_file()}
        names.update(self.members)
        return sorted(n for n in names if n.endswith(suffix) and not n.startswith(('raw-evidence/', 'system-snapshot/', 'method-snapshot/')))


def audit(batch, archive=None):
    evidence = Evidence(batch, archive)
    checked = {}; measurements = []; profiles = []; count = 0

    def verify(item):
        nonlocal count
        name = evidence.name(item['path'])
        payload = evidence.read(name)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item['sha256'] or ('bytes' in item and len(payload) != item['bytes']):
            raise ValueError('附件SHA或长度不同：' + name)
        if name in checked and checked[name] != digest:
            raise ValueError('附件身份冲突：' + name)
        checked[name] = digest; count += 1

    def link_times(directory, prefix, timings, repeats):
        nonlocal count
        actual = {key: [] for key in timings}
        for repeat in range(repeats):
            measurement = evidence.json(directory / (prefix + str(repeat)) / 'measurement.json')
            for record in measurement['records']:
                actual[record['key']].append(record['device_ms'])
        if actual != timings:
            raise ValueError('汇总计时与原始measurement不同：' + str(directory))
        count += 1

    for name in evidence.names('/probe.json'):
        probe = evidence.json(name)
        if probe.get('schema') != 'hgkda-device-interventions-v1':
            continue
        for case in probe['cases']:
            for launch in case['parent_launches']:
                for artifact in launch['artifacts']: verify(artifact)
            for variant in case['variants']:
                directory = PurePosixPath(name).parent / ('case-' + case['id']) / variant['name']
                source = directory / 'candidate.py'
                if hashlib.sha256(evidence.read(source)).hexdigest() != variant['source_sha256']:
                    raise ValueError('探针候选源码身份不同：' + str(source))
                for launch in variant.get('launches', []):
                    for artifact in launch['artifacts']: verify(artifact)
                if variant['status'] == 'passed':
                    link_times(directory, 'benchmark-', variant['timings'], 3)

    for name in evidence.names('/check.json'):
        gate = evidence.json(name)
        for index, confirmation in enumerate(gate.get('confirmations', [])):
            for key in ['parent_ir', 'candidate_ir']:
                for launch in confirmation[key]:
                    for artifact in launch['artifacts']: verify(artifact)
            prefix = 'benchmark-' + str(index // 2) + '-' + confirmation['dtype'] + '-'
            link_times(PurePosixPath(name).parent, prefix, confirmation['timings'], 5)

    for name in evidence.names('/measurement.json'):
        measurement = evidence.json(name)
        payload = evidence.read(measurement['csv_path'])
        if hashlib.sha256(payload).hexdigest() != measurement['csv_sha256']:
            raise ValueError('task_time原件身份不同：' + name)
        rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))[1:-1]
        if len(rows) != 16 or len(measurement['records']) != 2:
            raise ValueError('单kernel设备任务数不同：' + name)
        for index, record in enumerate(measurement['records']):
            tasks = rows[index * 8:(index + 1) * 8]
            if any(record['symbol'] not in row.get('kernel_name', '') for row in tasks):
                raise ValueError('设备任务符号不同：' + name)
            values = [float(row['task_time(us)']) for row in tasks[2:]]
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('设备时长无效：' + name)
            if values != record['active_device_us'] or not math.isclose(sum(values) / 6000, record['device_ms'], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('设备CSV无法重算：' + name)
            count += 1
        measurements.append(name)

    for name in evidence.names('/artifacts.json'):
        directory = PurePosixPath(name).parent
        if not any(token in directory.name for token in ['metrics', 'instruction']):
            continue
        for artifact in evidence.json(name): verify(artifact)
        identity = evidence.json(directory / 'identity.json')
        request = evidence.json(directory / 'request.json')
        response = evidence.json(directory / 'response.json')
        source = request['implementation']['sources'][0]['content']
        if hashlib.sha256(source.encode()).hexdigest() != identity['source_sha256'] or request['case_id'] != identity['case_id']:
            raise ValueError('profiling请求身份不同：' + name)
        facts = evidence.json(directory / 'facts.json')
        if facts['source_sha256'] != identity['source_sha256'] or facts['case_id'] != identity['case_id'] or facts['profile_status'] != response.get('status'):
            raise ValueError('profiling摘要身份或终态不同：' + name)
        count += 2
        profiles.append(dict(path=str(directory),status=response.get('status'),artifacts=len(evidence.json(name))))

    result = dict(status='passed', assertions=count, unique_artifacts=len(checked), device_measurements=len(measurements),
                  profiles=profiles, artifacts=checked,
                  scope='SHA/字节长度与设备CSV重算；不证明profiling因果结论，也不把静态IR计数当动态指令数')
    if evidence.tar: evidence.tar.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('核验输出已存在')
    result = audit(args.batch_dir, args.archive)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}, ensure_ascii=False))
