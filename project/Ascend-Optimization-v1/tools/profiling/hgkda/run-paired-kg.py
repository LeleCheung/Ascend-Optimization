#!/usr/bin/env python3
"""将冻结成对事实加载进KG角色，由CC继续优化，最终一次完整KGS验证。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex


def quote(command):
    return ' '.join(shlex.quote(str(x)) for x in command)


def main():
    root = Path('/data/hanle/ascend-optimization')
    tools = Path(__file__).resolve().parent
    s = importlib.util.spec_from_file_location('paired_kg_queue', tools/'run-nightly.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    queue = m.Queue('paired-pointwise-20261022', root, 'tle_yy')
    state_path = queue.batch/'kg-paired-state.json'
    if state_path.exists():
        raise ValueError('KG批次已有状态，不覆盖旧工作区')
    state = dict(status='queued', results={}, final_repeats=1,
        contribution='冻结成对硬件事实＋HG-KDA知识角色；CC/DeepSeek生成；独立语义和KGS验证')
    m.save(state_path, state)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            candidate_path = queue.batch/'candidate-state-v1.json'
            if not candidate_path.exists() or json.loads(candidate_path.read_text())['status'] != 'completed':
                raise RuntimeError('先完成独立候选队列，再新建KG任务')
            queue.idle()
            evidence = queue.batch/'paired-evidence.json'
            if not evidence.exists():
                queue.run('summarize-paired', quote([queue.py, tools/'summarize-paired.py',
                    '--batch-dir', queue.batch, '--output', evidence]), 600)
            overlay = root/'runtime'/queue.batch.name/'sources'
            if not overlay.exists():
                queue.run('build-paired-overlay', quote([queue.py, tools/'build-kg-overlay.py',
                    '--output', overlay, '--native-context', '--paired-evidence', evidence]), 600)
            for op in ['log10_', 'selu_']:
                queue.idle()
                state.update(status='running', current_operator=op)
                m.save(state_path, state)
                suffix = queue.batch.name+'-paired-hgkda'
                run_root = queue.runtime/'kg-runs'
                workspace = run_root/(op+'-'+suffix)
                output = queue.batch/'results'/op/'paired-HG-KDA'
                output.mkdir(parents=True, exist_ok=True)
                # Codex先验和修复明确作为seed来源记录，不称纯模型发现。
                result = json.loads(candidate_path.read_text())['results'].get(op, {})
                seed = (queue.batch/'candidates'/(op+'-rawptr-v1.py')
                        if result.get('status') == 'passed' else queue.batch/queue.records[op]['seed'])
                m.save(output/'input-provenance.json', dict(seed=str(seed),
                    seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                    paired_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    attribution='既有KG候选＋Codex成对诊断修复为起点；本组记录CC追加生成'))
                exports = dict(KG_SOURCE_ROOT=str(overlay), KG_RUN_ROOT=str(run_root),
                    KG_CATALOG_PATH=str(queue.batch/'catalog'), KG_WALLTIME_PORT='19659',
                    KG_PROFILE_PORT='19659', KG_MODEL='deepseek-flash')
                body = '\n'.join('export '+key+'='+shlex.quote(value) for key,value in exports.items())
                body += '\n'+quote(['bash', queue.project/'tools/evaluation/run-kg-operator-910b.sh',
                    op, 'profile', suffix, '2', seed])
                try:
                    queue.run('kg-paired-'+op, body, 10800)
                except RuntimeError:
                    process = workspace/'.kernelgen/run-process.json'
                    if not process.exists() or not json.loads(process.read_text()).get('finished_at'):
                        raise
                loaded = []
                for name in ['kernel-profile-analyzer.md', 'kernel-knowledge-profile-analyzer.md']:
                    for role in workspace.rglob(name):
                        raw = role.read_bytes()
                        if b'hgkda-paired-v1' not in raw:
                            continue
                        digest = hashlib.sha256(raw).hexdigest()
                        dest = output/('loaded-role-'+digest[:12]+'.md')
                        dest.write_bytes(raw)
                        loaded.append(dict(path=str(role), sha256=digest, archived=dest.name))
                m.save(output/'role-load-evidence.json', dict(roles=loaded,
                    overlay_manifest=json.loads((overlay/'hgkda-overlay.json').read_text())))
                if not loaded:
                    raise RuntimeError('没有找到实际加载的成对证据角色，不能声称已接入')
                for script, dest in [('export-kg-best.py', output/'export'),
                                     ('archive-kg-evidence.py', output/'native-evidence')]:
                    args = [queue.py, queue.project/'tools/evaluation'/script, workspace]
                    if script.startswith('export'):
                        args.append(op)
                    args.append(dest)
                    queue.run(script+'-paired-'+op, quote(args), 600)
                source = output/'export/main.py'
                semantic_path = output/'semantics.json'
                try:
                    queue.run('paired-kg-semantics-'+op, quote([queue.py, tools/'check-pointwise-semantics.py',
                        '--operator', op, '--source', source, '--output', semantic_path]), 1800)
                except RuntimeError:
                    if not semantic_path.exists():
                        raise
                semantic = json.loads(semantic_path.read_text())
                if semantic['status'] != 'passed':
                    state['results'][op] = dict(status='semantic_failed', source=str(source),
                        passed=semantic['passed'], total=semantic['total'])
                    m.save(state_path, state)
                    continue
                state['results'][op] = queue.evaluate(op, source, output/'formal', 'paired-HG-KDA', 1)
                m.save(state_path, state)
            state.update(status='completed', current_operator=None)
        except Exception as error:
            state.update(status='needs_attention', error=str(error))
            raise
        finally:
            m.save(state_path, state)


if __name__ == '__main__':
    main()
