"""从已评测 Native 快照补全诊断输入，禁止根据 case 序号猜测形状。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def enrich_native_context(context, snapshot):
    """只补充已通过身份核验的输入规格，不改原始测量或调用 KGS 评测。"""
    identity = snapshot['identity']
    if identity.get('adapter_kind') != 'native':
        return deepcopy(context)
    for name in ['round_num', 'solution_sha256', 'evaluation_fingerprint']:
        if context.get(name) != identity.get(name) or identity.get(name) is None:
            raise ValueError('Native 上下文身份不同：' + name)
    if canonical_sha256(snapshot['solution']) != identity.get('solution_sha256'):
        raise ValueError('Native 候选快照与身份不同')
    if canonical_sha256(snapshot['result']) != identity.get('result_sha256'):
        raise ValueError('Native 评测结果与身份不同')
    measured = snapshot['result'].get('per_workload', [])
    if context.get('per_workload') != measured:
        raise ValueError('Native 上下文测量与冻结结果不同')
    workloads = snapshot['workloads']
    names = [row['name'] for row in workloads]
    if (not names or len(set(names)) != len(names)
            or names != identity.get('workload_uuids')
            or [canonical_sha256(row) for row in workloads] != identity.get('workload_sha256')):
        raise ValueError('Native 用例或参数与冻结身份不同')
    rows = {row['uuid']: row for row in measured}
    if len(rows) != len(measured) or set(rows) != set(names):
        raise ValueError('Native 测量没有唯一覆盖冻结用例')
    profile_ids = context.get('profile_workload_uuids', [])
    if (profile_ids != identity.get('profile_workload_uuids')
            or len(set(profile_ids)) != len(profile_ids)
            or not set(profile_ids).issubset(names)):
        raise ValueError('Native profiling 用例与冻结身份不同')
    result = deepcopy(context)
    result['hgkda_workloads'] = {
        row['name']: {
            'inputs': deepcopy(row['inputs']),
            'seed': row.get('seed'),
            'tolerance': deepcopy(row.get('tolerance')),
            'phase': rows[row['name']].get('phase'),
            'profile_eligible': row['name'] in profile_ids,
            'workload_sha256': digest,
        }
        for row, digest in zip(workloads, identity['workload_sha256'])
    }
    result['hgkda_context_enrichment'] = {
        'schema_version': '1.0',
        'source': 'immutable_evaluation_snapshot',
        'solution_sha256': identity['solution_sha256'],
        'evaluation_fingerprint': identity['evaluation_fingerprint'],
        'instruction': '以 hgkda_workloads 的原始输入规格解释形状、dtype和参数；不由case序号猜测。',
    }
    return result
