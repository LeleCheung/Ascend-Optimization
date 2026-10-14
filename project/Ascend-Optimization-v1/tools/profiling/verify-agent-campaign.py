#!/usr/bin/env python3
"""离线核实完整优化链、两次最终评测及真机附件，兼容迁移后的绝对路径。"""
import argparse
import json
import math
from pathlib import Path
import urllib.parse

from optimization_campaign import checked, digest, parse_candidate
from ascend_profiling_workflow import check_artifacts, source_in


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_generation(directory, source_sha256):
    """新接口模式须有实际CC源码、请求适配和无思考输出的证据。"""
    matches = []
    for attempt, filename in ((1, 'claude-response.json'), (2, 'claude-repair-response.json')):
        response = directory / filename
        if not response.exists():
            continue
        raw = response.read_text(encoding='utf-8')
        try:
            proposal = parse_candidate(raw)
        except (ValueError, KeyError, SyntaxError, TypeError):
            continue
        if digest(proposal['source'].replace('\r\n', '\n')) != source_sha256:
            continue
        requests = read(directory / f'provider-request-{attempt}.json')['requests']
        messages = [row for row in requests if urllib.parse.urlsplit(row['route']).path == '/v1/messages']
        if (not messages or any(row.get('forced_thinking_type') != 'disabled' for row in messages)
                or not any(row.get('http_status') == 200 for row in messages)):
            raise ValueError('缺少显式关闭供应商思考的成功请求')
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            for block in event.get('message', {}).get('content', []):
                if isinstance(block, dict) and block.get('thinking'):
                    raise ValueError('供应商关闭模式仍输出思考')
        matches.append(attempt)
    if not matches:
        raise ValueError('候选无法对应实际CC生成的完整源码')
    return matches[-1]


def verify(root):
    state = read(root / 'campaign-state.json')
    if state['status'] != 'completed':
        raise ValueError('campaign 尚未完成')
    seed_path = root / 'seed-evaluation/seed-1.result.json'
    contract, best_score = checked(seed_path)
    configuration = state.get('configuration', {})
    if configuration.get('timing_scope', 'device_kernel') != contract['timing_scope']:
        raise ValueError('配置与实际计时范围不同')
    # 早期批次尚未保存知识全文；新格式必须验证快照，不能补造历史文件。
    if configuration.get('knowledge_sha256') and (configuration.get('knowledge_snapshot_required') or (root / 'knowledge.md').exists()):
        if not (root / 'knowledge.md').is_file() or digest((root / 'knowledge.md').read_text(encoding='utf-8')) != configuration['knowledge_sha256']:
            raise ValueError('知识快照SHA不匹配')
    best_digest = contract['source_sha256']
    if best_digest != state['seed_sha256']:
        raise ValueError('起点源码 SHA 不一致')
    rounds, artifacts = set(), 0
    for row in state['history']:
        index = row['round']
        if index in rounds:
            raise ValueError('重复优化轮次')
        rounds.add(index)
        directory = root / f'round-{index:02d}'
        if not math.isclose(row['parent_score'], best_score, rel_tol=1e-10):
            raise ValueError('历史起点分数不一致')
        diagnosis_path = directory / 'profiles/diagnosis.json'
        if diagnosis_path.exists():
            diagnosis = read(diagnosis_path)
            if (diagnosis['candidate']['sha256'] != best_digest
                    or diagnosis['evaluation']['benchmark_fingerprint'] != contract['fingerprint']
                    or diagnosis['timing_scope'] != contract['timing_scope']):
                raise ValueError('诊断未绑定当轮最佳候选或合同')
            source = None
            for profile in sorted((directory / 'profiles').glob('case-*')):
                response_path = profile / 'response.json'
                if not response_path.exists():
                    continue
                response, request = read(response_path), read(profile / 'request.json')
                if response.get('status') != 'completed':
                    continue
                source = source_in(request)
                if (digest(source) != best_digest or request['binding'] != contract['binding']
                        or request['benchmark_fingerprint'] != contract['fingerprint']
                        or request['case_id'] not in contract['timing']
                        or response['workload_name'] != request['case_id']):
                    raise ValueError('profile 与当轮候选、case 或合同不匹配')
                artifacts += len(check_artifacts(response, profile / 'artifacts'))
        if row['status'] == 'REPLAYED':
            if (state.get('workflow_kind') != 'historical-replay'
                    or row.get('source_sha256') != best_digest or not diagnosis_path.exists()
                    or not (directory / '给CC的提示词.md').exists()):
                raise ValueError('历史复现轮缺少身份、源码绑定或方法记录')
        if row['status'] != 'PASSED':
            continue
        result, score = checked(directory / 'evaluation/trial-1.result.json', contract)
        if (result['source_sha256'] != row['source_sha256']
                or digest((directory / 'candidate.py').read_text(encoding='utf-8')) != row['source_sha256']
                or not math.isclose(score, row['score'], rel_tol=1e-10)):
            raise ValueError('候选源码、结果或分数不一致')
        if configuration.get('provider_thinking') == 'disabled':
            verify_generation(directory, row['source_sha256'])
        accepted = score > best_score * 1.005
        if bool(row['accepted']) != accepted:
            raise ValueError('最佳选择不符合0.5%门槛')
        if accepted:
            best_score, best_digest = score, result['source_sha256']
    if best_digest != state['final_source_sha256'] or not math.isclose(best_score, state['best']['score'], rel_tol=1e-10):
        raise ValueError('最终选择未对应历史最佳')
    if digest((root / 'final.py').read_text(encoding='utf-8')) != best_digest:
        raise ValueError('最终源码 SHA 不一致')
    scores = []
    for index in (1, 2):
        result, score = checked(root / 'final-evaluation' / f'final-{index}.result.json', contract)
        if result['source_sha256'] != best_digest or not math.isclose(score, state['final_scores'][index - 1], rel_tol=1e-10):
            raise ValueError('两次复验未绑定同一最终源码或分数')
        scores.append(score)
    if not math.isclose(min(scores), state['final_score'], rel_tol=1e-10):
        raise ValueError('最终成绩不是两次复验的较慢一次')
    if not artifacts:
        raise ValueError('缺少真机原始采集附件')
    return {'operator': state['operator'], 'status': 'VERIFIED', 'rounds': len(rounds),
            'full_tests': len(contract['rows']), 'artifact_count': artifacts,
            'final_scores': scores, 'source_sha256': best_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.campaign), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
