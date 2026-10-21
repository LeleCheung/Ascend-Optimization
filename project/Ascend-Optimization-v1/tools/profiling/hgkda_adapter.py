"""可选的 HG-KDA 分阶段策略；共用原生 KGS 采集与完整评测。"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / 'hgkda/skills/hgkda'
spec = importlib.util.spec_from_file_location('hgkda_diagnose', ROOT / 'scripts/diagnose.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def fingerprint():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return hashlib.sha256(json.dumps([
        (p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in files], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def instruction_hotspots(listing, source, *, count=12):
    """只汇总源码中 jit 函数对应的指令热点，避免将输入构建 kernel 当作候选。"""
    symbols = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
               and any('jit' in ast.unparse(d) for d in n.decorator_list)}
    selected = []
    for kernel in listing.get('kernels', []):
        name = kernel.get('name', '')
        if not any(name == s or name.startswith(s + '_') for s in symbols):
            continue
        for core in kernel.get('cores', []):
            rows = [r for r in core.get('instructions', []) if isinstance(r, dict)]
            top = sorted(rows, key=lambda r: (r.get('running_time_us') or 0,
                                              r.get('cycles') or 0), reverse=True)[:count]
            selected.append({'kernel': name, 'core': core.get('name'),
                             'listed_instruction_count': core.get('listed_instruction_count'),
                             'mapped_instruction_count': core.get('mapped_instruction_count'),
                             'hotspots': top})
    return {'scope': '模拟器单指令观测，不能相加为真机总耗时', 'kernels': selected,
            'status': 'matched' if selected else 'no_candidate_kernel_match'}


def render_staged(report):
    lines = ['\n## HG-KDA 分阶段体检', '',
             '流水比例用于提出检查方向；当前未建立数值 Roofline。']
    for case in report.get('cases', []):
        item = case.get('hgkda', {})
        lines.extend(['', '### ' + case['case_id'], '',
                      '算子类型：' + item.get('operator_type', 'unknown') +
                      '；阶段：' + item.get('stage', 'unknown') + '。'])
        for hypothesis in item.get('hypotheses', []):
            lines.append('- ' + hypothesis['metric'] + '=' + str(hypothesis['value']) +
                         '：' + hypothesis['action'] + '。')
        for collection in item.get('requested_collections', []):
            lines.append('- 待采集 ' + collection['level'] + '：' + collection['reason'] + '。')
        for limitation in item.get('limitations', []):
            lines.append('- ' + limitation + '。')
        if item.get('knowledge'):
            lines.extend(['', '知识入口：' + '、'.join(item['knowledge']) + '。'])
    return '\n'.join(lines) + '\n'


def check_supplementary(operator, candidate, directory, *, run):
    if operator != 'rad2deg_':
        return None
    result_path = directory / 'supplementary-semantics.json'
    if not result_path.exists():
        run([sys.executable, str(ROOT / 'scripts/check_rad2deg.py'),
             '--source', str(candidate), '--output', str(result_path)],
            directory / 'supplementary-semantics.log', timeout=600)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if (result.get('status') != 'PASSED' or len(result.get('cases', [])) != 21
            or result.get('source_sha256') != hashlib.sha256(candidate.read_bytes()).hexdigest()):
        raise ValueError('补充原地语义检查未完整通过或源码身份不匹配')
    return str(result_path)


def analyze_staged(report, evaluation, directory, server, *, workflow, run, assert_idle, call):
    service = call(server, '/status', timeout=30)
    policy.enrich(report, service)
    # 每轮最多升级一个代表 case。其余建议保留，避免多次昂贵模拟器重放。
    upgraded = False
    for case in report['cases']:
        requests = case['hgkda']['requested_collections']
        if upgraded or not any(x['level'] == 'instruction' for x in requests):
            continue
        metric = case['profiles'].get('metrics')
        if metric is None:
            continue
        upgraded = True
        # KGS case_id 可包含 /、::，不能直接作为跨平台证据文件名。
        case_key = hashlib.sha256(case['case_id'].encode()).hexdigest()[:20]
        output = directory / 'instruction' / ('case-' + case_key)
        response_path = output / 'response.json'
        if output.exists() and not response_path.exists():
            raise RuntimeError('指令请求已有目录但结果未知，停止自动重发：' + str(output))
        try:
            if not response_path.exists():
                assert_idle()
                request_path = Path(metric['request'])
                run([sys.executable, str(Path(__file__).resolve().parents[1] / 'evaluation/profile-operator.py'),
                     '--evaluation', str(evaluation),
                     '--request', str(evaluation.with_name(evaluation.name.replace('.result.json', '.request.json'))),
                     '--inspect', str(request_path.parent / 'inspect.json'),
                     '--case-id', case['case_id'], '--output', str(output),
                     '--server', server, '--level', 'instruction', '--timeout', '900'],
                    directory / ('instruction-' + case_key + '.log'), timeout=1200)
            response = json.loads(response_path.read_text(encoding='utf-8'))
            request = json.loads((output / 'request.json').read_text(encoding='utf-8'))
            metric_request = json.loads(Path(metric['request']).read_text(encoding='utf-8'))
            if (response.get('status') != 'completed' or request['case_id'] != case['case_id']
                    or request['implementation'] != metric_request['implementation']
                    or request['binding'] != metric_request['binding']
                    or request['benchmark_fingerprint'] != metric_request['benchmark_fingerprint']):
                raise ValueError('指令采集状态或候选身份不匹配')
            artifacts = workflow.check_artifacts(response, output / 'artifacts')
            listings = [a for a in artifacts if a['kind'] == 'instruction_listing']
            source = workflow.source_in(request)
            case['profiles']['instruction'] = {
                'status': response['status'],
                'profile_id': response.get('profile_id'), 'response': str(response_path),
                'request': str(output / 'request.json'), 'artifacts': artifacts,
                'instruction_count': response.get('summary', {}).get('simulator_instruction_count'),
                'kernel': workflow.select_kernel(response, report['operator']),
                'candidate_hotspots': [{'path': a['path'], 'analysis': instruction_hotspots(
                    json.loads(Path(a['path']).read_text(encoding='utf-8-sig')), source)}
                    for a in listings[:1]],
            }
        except Exception as error:
            # 请求中断、服务繁忙等异常由上层终止，不用模拟器失败掩盖设备状态未知。
            if not response_path.exists():
                raise
            case['instruction_error'] = str(error)
    policy.enrich(report, service)
    report['routed_knowledge'] = policy.knowledge_for(report)
    return report
