"""只将完整评测及补测均通过的原生实现加入补充对照。"""
from pathlib import Path


def native_sources(state, operator):
    sources = []
    native = state.get('native_candidates', {}).get(operator) or {}
    if qualified(native):
        sources.append(('HG-KDA-AscendC-v1', Path(native['source'])))
    prefix = 'queued-proposal:' + operator + ':'
    for key, job in sorted(state.get('jobs', {}).items()):
        result = job.get('result') or {}
        if (key.startswith(prefix) and job.get('status') == 'completed'
                and 'AscendC' in result.get('backend', '') and qualified(result)):
            sources.append(('HG-KDA-' + key[len(prefix):], Path(result['source'])))
    return sources


def qualified(record):
    return (record.get('status') == 'PASSED'
            and record.get('supplementary_status') == 'passed'
            and record.get('num_passed') == record.get('num_workloads')
            and bool(record.get('num_workloads')))
