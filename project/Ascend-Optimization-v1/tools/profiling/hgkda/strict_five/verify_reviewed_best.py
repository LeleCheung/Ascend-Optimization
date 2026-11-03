"""等待搜索终态后核验人工审查的新最佳；复用正式成绩，补测与实际编译分开留证。"""
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
BATCH = PROJECT/'operators/campaigns/strict-five-20261103'
PY = '/usr/local/python3.11.15/bin/python3.11'
DIGEST = '8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)


def main():
    sys.path.insert(0, str(HERE.parent/'strict_night'))
    from admission import static
    from boundary_upgrade import terminal_workspace
    from expm1_exact_probe import qualified
    from measurements import score
    spec = importlib.util.spec_from_file_location('strict_five_review_queue', HERE.parent/'run-nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = BATCH/'reviewed-best-process.json'
    assert not process.exists(), '已有人工核验身份，不重复启动'
    identity = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                    status='waiting_for_optimizer')
    save(process, identity)
    result_path = BATCH/'reviewed-best-result.json'
    result = dict(source_sha256=DIGEST, CANN_accepted=False)
    try:
        while read(BATCH/'optimizer-process.json')['status'] in {'waiting_for_project_lock','running'}:
            time.sleep(10)
        assert read(BATCH/'optimizer-process.json')['status']=='completed'
        with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert all(terminal_workspace(w) for w in (ROOT/'runs/strict-five-20261103/kg-runs').iterdir() if w.is_dir())
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            state = read(BATCH/'search-state.json')
            best = dict(state['operators']['expm1_'].get('best') or {})
            if best.get('source_sha256') != DIGEST or best.get('qualification') != 'passed':
                result.update(status='not_selected_or_supplement_failed', selected=best.get('source_sha256'))
            else:
                source = Path(best['source'])
                assert sha(source) == DIGEST and not static(source.read_text())
                manual = read(BATCH/'reviews/expm1_'/ (DIGEST+'.json'))
                assert manual['source_sha256']==DIGEST and manual['status']=='reviewed_strict_triton'
                assert manual['all_branches_reviewed'] and manual['dependencies_reviewed']
                assert not (source.parent/'review.json').exists()
                save(source.parent/'review.json', manual)
                # KG的完整正式结果已存在；只核验逐case计算与补测，不提交第二份正式计时。
                value = read(best['search_result'])
                current = score(value)
                supplement = read(best['extras']['pytorch']['path'])
                assert supplement['source_sha256']==DIGEST and supplement['status']=='passed'
                identity.update(status='running', started_at=module.now())
                save(process, identity)

                class Host:
                    def cmd(self, key, args, budget=2100):
                        queue.idle()
                        return queue.run('reviewed-best-'+key, shlex.join([str(v) for v in args]), budget)
                runner = Host()
                exact_tool = HERE.parent/'strict_night/expm1_exact_probe.py'
                frozen = source.parent/('exact-expm1-checker-'+sha(exact_tool)+'.py')
                assert not frozen.exists()
                frozen.write_bytes(exact_tool.read_bytes())
                exact = source.parent/'exact-expm1-probe'
                runner.cmd('exact', [PY, frozen, '--source', source, '--output', exact], 900)
                result.update(PyTorch=current, source=str(source), formal=best['search_result'],
                    supplement=best['extras']['pytorch']['path'], exact=str(exact/'result.json'))
                if qualified(read(exact/'result.json'), DIGEST):
                    domain_tool = HERE/'probe_half_domain.py'
                    domain_frozen = source.parent/('half-domain-checker-'+sha(domain_tool)+'.py')
                    assert not domain_frozen.exists()
                    domain_frozen.write_bytes(domain_tool.read_bytes())
                    domain = source.parent/'full-fp16-domain'
                    runner.cmd('domain', [PY, domain_frozen, '--source', source, '--output', domain], 1200)
                    result['full_fp16_domain'] = str(domain/'result.json')
                    if read(domain/'result.json')['status']=='passed':
                        final_spec = importlib.util.spec_from_file_location('strict_five_best_finalize', HERE.parent/'strict_night/finalize.py')
                        final = importlib.util.module_from_spec(final_spec)
                        final_spec.loader.exec_module(final)
                        final.BATCH = BATCH
                        final.verify(runner, 'expm1_', best)
                        save(BATCH/'reviewed-best-candidate.json', best)
                        result.update(status='verified_original_PyTorch_only', backend=best['compiled_status'])
                    else:
                        result['status']='failed_full_fp16_domain'
                else:
                    result['status']='failed_exact_supplement'
            save(result_path, result)
            queue.idle()
            identity.update(status='completed', finished_at=module.now(), result=str(result_path))
            save(process, identity)
    except Exception as error:
        identity.update(status='failed', error=type(error).__name__+': '+str(error), finished_at=module.now())
        save(process, identity)
        result.update(status='requires_review', error=identity['error'])
        save(result_path, result)
        raise


if __name__ == '__main__':
    main()
