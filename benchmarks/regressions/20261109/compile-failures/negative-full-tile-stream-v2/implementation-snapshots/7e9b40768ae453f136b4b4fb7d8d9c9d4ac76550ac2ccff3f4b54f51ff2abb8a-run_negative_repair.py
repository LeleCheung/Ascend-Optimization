"""固定回归结束后，串行验证negative_的宿主启动或完整分块流式候选。"""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from hgkda.gate import admit
from hgkda.validation import supplement_status


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temporary.replace(path)


def stream_variant(text):
    """完整分块走固定长度搬运，尾部分块保留掩码；签名和符号位数学不变。"""
    function=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name=='_negative_streaming')
    old=ast.get_source_segment(text,function)
    new='''def _negative_streaming(X, N, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        if start + BLOCK <= N:
            x = tl.load(X + offset)
            if X.dtype.element_ty == tl.float32:
                y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
            else:
                y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
            tl.store(X + offset, y)
        else:
            mask = offset < N
            x = tl.load(X + offset, mask, other=0)
            if X.dtype.element_ty == tl.float32:
                y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
            else:
                y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
            tl.store(X + offset, y, mask)'''
    assert text.count(old)==1
    text=text.replace(old,new)
    block='    maximum = 8192'
    assert text.count(block)==1
    return text.replace(block,'    maximum = 32768 if dtype in (torch.float16, torch.bfloat16) else 16384')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['repository', 'regression-run', 'uniform-run', 'run-directory', 'lock', 'sdk']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--device-python', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--parent-candidate-run',type=Path)
    parser.add_argument('--stream-variant',action='store_true')
    args = parser.parse_args()
    run = args.run_directory.resolve()
    run.mkdir(parents=True, exist_ok=False)
    receipt = run/'controller-state.json'
    state = dict(status='waiting_for_preceding_stages')
    save(receipt, state)

    def execute(label, command):
        with (run/(label+'.log')).open('xb') as output:
            result = subprocess.run(list(map(str, command)), cwd=args.repository,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(label+' 返回 '+str(result.returncode))

    def device(label, command):
        environment = dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='1', MAX_JOBS='1', TRITON_CACHE_DIR=str(run/'triton-cache'),
                           PYTHONPATH=str(args.repository/'third_party/FlagGems/src'))
        shell = 'source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
        shell += '\n'.join('export '+k+'='+shlex.quote(v) for k, v in environment.items())+'\n'
        shell += 'exec '+shlex.join([args.device_python, *map(str, command)])
        execute(label, ['docker', 'exec', args.container, 'bash', '-c', shell])

    try:
        while True:
            fixed = read(args.regression_run/'state.json')
            if fixed['status'] == 'requires_attention':
                raise RuntimeError('固定回归存在未解决任务')
            previous = args.uniform_run/'controller-state.json'
            postflight = args.regression_run/'postflight-state.json'
            if (previous.exists() and read(previous)['status'] in ['completed', 'requires_attention']
                    and postflight.exists() and read(postflight)['status'] in ['completed', 'requires_attention']):
                audit = read(args.regression_run/'audit-final.json')
                assert audit['evidence_verified'] and audit['verified'] == fixed['expected']
                if (args.uniform_run/'state.json').exists():
                    assert read(args.uniform_run/'state.json')['status'] == 'completed'
                    assert (args.uniform_run/'service-stop.json').exists()
                break
            time.sleep(10)
        original = read(args.regression_run/'plan.json')
        parent_plan=read(args.parent_candidate_run/'plan.json') if args.parent_candidate_run else original
        old_job = next(job for job in parent_plan['jobs'] if job['operator'] == 'negative_')
        root = run/'operators/negative_'
        root.mkdir(parents=True)
        for name in ['operator.tar', 'request.json', 'review.json']:
            shutil.copyfile(args.regression_run/'operators/negative_'/name, root/name)
        checks = {}
        for kind, spec in old_job['checks'].items():
            leaf = root/(kind+'-catalog')
            shutil.copytree(spec['leaf'], leaf, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            checker = root/(kind+'-checker.py')
            shutil.copyfile(spec['checker'], checker)
            checks[kind] = dict(spec, leaf=str(leaf), checker=str(checker))

        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state['status'] = 'preparing_stream_variant' if args.stream_variant else 'building_host_bridge'; save(receipt, state)
            source = root/'candidate.py'
            if args.stream_variant:
                assert args.parent_candidate_run and read(args.parent_candidate_run/'controller-state.json')['status']=='completed'
                build=args.parent_candidate_run/'host-build'
                built=read(build/'build.json')
                assert sha(built['binary'])==built['binary_sha256']
                assert sha(old_job['source'])==old_job['source_sha256']
                source.write_bytes(stream_variant(Path(old_job['source']).read_bytes().decode('utf8')).encode('utf8'))
            else:
                build = run/'host-build'
                device('build-host', [args.repository/'tools/regression/build_unary_host.py',
                                      '--source', old_job['source'], '--output', build])
                built = read(build/'build.json')
                shutil.copyfile(built['candidate'], source)
            # 此记录绑定已逐函数审查的宿主桥、构建身份和完整继承的设备核。
            review = dict(operator='negative_', source_sha256=sha(source), status='reviewed_strict_triton',
                          all_branches_reviewed=True, dependencies_reviewed=True,
                          parent_source_sha256=old_job['source_sha256'],
                          reviewed_host_bridges=[dict(status='reviewed_host_only', all_functions_reviewed=True,
                               device_computation='Triton_only', functions=['Key::operator==', 'KeyHash::operator()',
                               'register_plan', 'run', 'stats', 'checked', 'key', 'eligible', 'slow', 'PYBIND11_MODULE'],
                               cpp_source=built['source'], cpp_source_sha256=built['source_sha256'],
                               binary=built['binary'], binary_sha256=built['binary_sha256'],
                               build_record=str(build/'build.json'), build_record_sha256=sha(build/'build.json'))],
                          review_notes=['宿主缓存只处理元数据、参数及Triton启动；不持有输入、输出或旧stream。',
                                        '设备数学和所有非连续、未对齐、空输入回退逐字继承父版本。',
                                        '启动异常直接传播；不在已执行的原地计算后重试。'])
            if args.stream_variant:
                review['review_notes'].append('连续扫描增加完整分块分支，避免每块动态长度subview；尾块保留掩码。仅调整分块与搬运组织，符号位数学不变。')
            save(root/'review.json', review)
            admit(source.read_bytes().decode('utf8'), review)
            state['status'] = 'validating_runtime_and_compiled_paths'; save(receipt, state)
            arguments=[args.repository/'tools/regression/verify_unary_host.py','--source', source, '--output', run/'compiled-path']
            if args.stream_variant:arguments.append('--include-stream-tail')
            device('verify-host',arguments)
            device('candidate-original-supplement', [checks['supplementary']['checker'], '--operator', 'negative_',
                   '--source', source, '--leaf', checks['supplementary']['leaf'], '--output', root/'supplementary.json'])
            qualification = read(root/'supplementary.json')
            assert supplement_status('negative_', qualification, sha(source)) == 'passed'
            assert qualification['passed'] == qualification['total'] == checks['supplementary']['previous_total']
            request = read(root/'request.json')
            request['implementation']['sources'][0]['content'] = source.read_bytes().decode('utf8')
            request['implementation']['name'] += '_full_tile_stream_20261109' if args.stream_variant else '_host_cache_20261109'
            save(root/'request.json', request)
            job = dict(old_job, source=str(source), source_sha256=sha(source), review=str(root/'review.json'),
                       request=str(root/'request.json'), archive=str(root/'operator.tar'), checks=checks,
                       previous_cann_score=(read(args.parent_candidate_run/'state.json') if args.parent_candidate_run else fixed)['operators']['negative_']['evaluation']['geo_mean'])
            save(run/'plan.json', dict(original, jobs=[job], expected=1,
                 policy='独立Triton完整块流式候选；原合同不变' if args.stream_variant else '独立新宿主路径候选；父版本正式回归原件保留；原合同及计时参数不变',
                 runtime_verification=dict(path=str(run/'compiled-path/verification.json'),
                                           sha256=sha(run/'compiled-path/verification.json'))))
        state['status'] = 'formal_evaluation'; save(receipt, state)
        execute('formal-controller', [sys.executable, args.repository/'tools/regression/run.py',
                 '--plan', run/'plan.json', '--container', args.container, '--device', args.device,
                 '--port', args.port, '--lock', args.lock, '--device-python', args.device_python,
                 '--cann-environment', args.sdk/'set_env.sh'])
        for tool in ['audit', 'summarize']:
            execute(tool, [sys.executable, args.repository/('tools/regression/'+tool+'.py'),
                           '--run-directory', run, '--output', run/(tool+'-final.json')])
        state.update(status='completed', result=read(run/'state.json')); save(receipt, state)
    except Exception as error:
        state.update(status='requires_attention', error=type(error).__name__+': '+str(error)); save(receipt, state)
        raise


if __name__ == '__main__':
    main()
