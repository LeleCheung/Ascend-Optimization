"""新稿先通过离线编译，再在原项目卡锁内一次正式评测及补测。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'
PY = '/usr/local/python3.11.15/bin/python3.11'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', default='codex-fma-series-v1')
    parser.add_argument('--compile-label', default='compile-fma-v1')
    parser.add_argument('--identity-label', default='fma-evaluator')
    parser.add_argument('--proposal-root', choices=['proposals/expm1_', 'compile-probes'], default='proposals/expm1_')
    parser.add_argument('--after-optimizer', action='store_true')
    args = parser.parse_args()
    assert all(value and set(value) <= set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
               for value in [args.proposal, args.compile_label, args.identity_label])
    sys.path.insert(0, str(HERE.parent / 'strict_night'))
    from admission import static
    from measurements import score, computation_identity
    from signature_gate import admit
    from expm1_exact_probe import qualified
    spec = importlib.util.spec_from_file_location('strict_next_fma_queue', HERE.parent / 'run-nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity_path = BATCH / (args.identity_label + '-process.json')
    assert not identity_path.exists(), '已有提案请求身份，不能重复启动'
    identity = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    status='waiting_for_compile_and_profiles')
    save(identity_path, identity)
    folder = BATCH / args.proposal_root / args.proposal
    source = folder / 'main.py'
    compiled = BATCH / 'analysis/expm1_' / args.compile_label / 'result.json'
    try:
        while not compiled.exists() or read(compiled)['status'] == 'in_progress':
            time.sleep(10)
        assert read(compiled)['status'] == 'compiled' and read(compiled)['source_sha256'] == sha(source)
        while read(BATCH / 'collector-process.json')['status'] in ['waiting_for_project_lock', 'running']:
            time.sleep(10)
        assert read(BATCH / 'collector-process.json')['status'] == 'completed'
        if args.after_optimizer:
            identity['status'] = 'waiting_for_optimizer'
            save(identity_path, identity)
            path = BATCH / 'optimizer-process.json'
            while read(path)['status'] in ['waiting_for_proposal_and_profiles', 'running']:
                time.sleep(10)
            value = read(path)
            assert value['status'] == 'completed', '优化器异常，先核对其设备请求与工作区'
            process = Path('/proc') / str(value['pid']) / 'stat'
            while process.exists() and process.read_text().rsplit(')', 1)[1].split()[19] == str(value['process_start']):
                time.sleep(2)
        with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            identity.update(status='running', started_at=module.now())
            save(identity_path, identity)
            class Host:
                def cmd(self, key, args, budget=2100):
                    queue.idle()
                    return queue.run(folder.name + '-' + key, shlex.join([str(v) for v in args]), budget)
            runner = Host()
            leaf = BATCH / 'catalog/ops/expm1_'
            review = read(folder / 'review.json')
            assert review['source_sha256'] == sha(source) and review['all_branches_reviewed'] and review['dependencies_reviewed']
            history = read(folder / 'history-admission.json')
            assert history['source_sha256'] == sha(source) and not history['duplicates']
            assert history['status'] == 'new_computation_identity'
            # 排队期间KG可能已经生成同稿；取得卡锁后重新核对实际工作区，不能只信启动时历史。
            normalized = computation_identity(source)
            duplicates = []
            for path in (ROOT / 'runs').glob('*/kg-runs/expm1_*/stages/optimize/work/.kernelgen/evals/round-*/main.py'):
                try:
                    same = computation_identity(path) == normalized
                except SyntaxError:
                    same = False
                if same:
                    duplicates.append(dict(source=str(path), result=str(path.with_name('result.json')),
                                           result_exists=path.with_name('result.json').exists()))
            if duplicates:
                save(folder / 'result.json', dict(status='existing_computation_requires_review',
                     source_sha256=sha(source), duplicates=duplicates, device_request_submitted=False,
                     reason='先复用或核对排队期间已有的同计算身份结果，不重复正式评测'))
                identity.update(status='completed', finished_at=module.now())
                save(identity_path, identity)
                return
            assert not static(source.read_text())
            signature = admit(source, leaf / 'definition.json')
            save(folder / 'signature-admission.json', signature)
            assert signature['status'] == 'admitted'
            result = dict(status='running', source_sha256=sha(source), reference='original_PyTorch_only', CANN_accepted=False)
            save(folder / 'result.json', result)
            runner.cmd('fma-v1-runtime', [PY, HERE.parent / 'strict_triton/runtime_audit.py', '--operator', 'expm1_',
                       '--source', source, '--leaf', leaf, '--output', folder / 'runtime-audit.json'], 1200)
            assert read(folder / 'runtime-audit.json')['status'] == 'passed'
            runner.cmd('fma-v1-formal', [PY, HERE.parent / 'cann_priority/evaluate.py', '--operator', 'expm1_',
                       '--source', source, '--leaf', leaf, '--output', folder / 'pytorch', '--server', queue.server,
                       '--label', 'strict-PyTorch', '--kind', 'strict_Triton_Codex_proposal'], 2100)
            formal = folder / 'pytorch/strict-PyTorch-1.result.json'
            value = read(formal)
            result.update(formal=str(formal), formal_status=value['status'])
            if value['status'] == 'PASSED' and not value.get('is_hack'):
                current = score(value)
                result['PyTorch'] = current
                extra = folder / 'extra-pytorch.json'
                runner.cmd('fma-v1-extra', [PY, HERE.parent / 'cann_priority/validate_cann.py', '--operator', 'expm1_',
                           '--source', source, '--leaf', leaf, '--output', extra], 1200)
                exact_tool = HERE.parent / 'strict_night/expm1_exact_probe.py'
                exact_frozen = folder / ('exact-checker-' + sha(exact_tool) + '.py')
                assert not exact_frozen.exists()
                exact_frozen.write_bytes(exact_tool.read_bytes())
                exact = folder / 'exact-expm1-probe'
                runner.cmd('fma-v1-exact', [PY, exact_frozen, '--source', source, '--output', exact], 900)
                result.update(supplement=str(extra), exact=str(exact / 'result.json'))
                passed = read(extra)['status'] == 'passed' and qualified(read(exact / 'result.json'), sha(source))
                result.update(status='evaluated' if passed else 'supplement_failed', qualified_original=passed)
                if passed:
                    domain_tool = HERE.parent / 'strict_five/probe_half_domain.py'
                    domain_frozen = folder / ('domain-checker-' + sha(domain_tool) + '.py')
                    domain_frozen.write_bytes(domain_tool.read_bytes())
                    domain = folder / 'full-fp16-domain'
                    runner.cmd('fma-v1-domain', [PY, domain_frozen, '--source', source, '--output', domain], 1200)
                    result['full_fp16_domain'] = str(domain / 'result.json')
                    if read(domain / 'result.json')['status'] != 'passed':
                        result.update(status='domain_failed', qualified_original=False)
                    elif current > .6465574073403593:
                        best = dict(source=str(source), source_sha256=sha(source), search_result=str(formal), search_score=current,
                                    qualification='passed', review_status='reviewed_strict_triton',
                                    reference='original_PyTorch_only', pytorch_result=str(formal), kind='Codex_proposal',
                                    provenance=read(folder / 'spec.json')['attribution'],
                                    extras={'pytorch': dict(path=str(extra), status='passed')})
                        final_spec = importlib.util.spec_from_file_location('strict_next_fma_backend', HERE.parent / 'strict_night/finalize.py')
                        final = importlib.util.module_from_spec(final_spec)
                        final_spec.loader.exec_module(final)
                        final.BATCH = BATCH
                        final.verify(runner, 'expm1_', best)
                        save(folder / 'verified-candidate.json', best)
                        result.update(improved=True, backend=best['compiled_status'])
                    else:
                        result.update(improved=False, reason='低于冻结最佳，保留一次正式结果，不重测挑分')
            else:
                result['status'] = 'formal_failed'
            save(folder / 'result.json', result)
            queue.idle()
            identity.update(status='completed', finished_at=module.now())
            save(identity_path, identity)
    except Exception as error:
        identity.update(status='requires_review', error=type(error).__name__ + ': ' + str(error), finished_at=module.now())
        save(identity_path, identity)
        raise


if __name__ == '__main__':
    main()
