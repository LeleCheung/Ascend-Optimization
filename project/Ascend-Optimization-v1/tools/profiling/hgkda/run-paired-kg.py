#!/usr/bin/env python3
"""将冻结成对事实加载进KG角色，由CC继续优化，最终一次完整KGS验证。"""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import time


def quote(command):
    return ' '.join(shlex.quote(str(x)) for x in command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='paired-pointwise-20261022')
    parser.add_argument('--seed-plan', action='store_true', help='直接采用本批冻结seed，不要求旧独立候选队列')
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='仅恢复已记录异常的控制队列；不重启已有KG worker')
    config = parser.parse_args()
    if not config.batch.replace('-', '').replace('_', '').isalnum() or config.rounds < 2:
        raise ValueError('非法批次名或轮数')
    root = Path('/data/hanle/ascend-optimization')
    tools = Path(__file__).resolve().parent
    s = importlib.util.spec_from_file_location('paired_kg_queue', tools/'run-nightly.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    queue = m.Queue(config.batch, root, 'tle_yy')
    state_path = queue.batch/'kg-paired-state.json'
    if state_path.exists():
        if not config.resume:
            raise ValueError('KG批次已有状态，不覆盖旧工作区')
        state = json.loads(state_path.read_text())
        if state['status'] != 'needs_attention':
            raise ValueError('只能恢复已停止的异常队列')
        backup = queue.batch/('kg-state-before-resume-'+str(time.time_ns() if hasattr(time, 'time_ns') else int(time.time()*1e9))+'.json')
        backup.write_bytes(state_path.read_bytes())
        state.pop('error', None)
    else:
        if config.resume:
            raise ValueError('不存在可恢复的状态')
        state = dict(status='queued', results={}, final_repeats=1,
            contribution='冻结成对硬件事实＋HG-KDA知识角色；CC/DeepSeek生成；独立语义和KGS验证')
    m.save(state_path, state)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            candidate_path = queue.batch/'candidate-state-v1.json'
            if not config.seed_plan and (not candidate_path.exists() or json.loads(candidate_path.read_text())['status'] != 'completed'):
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
                if op in state['results']:
                    continue
                queue.idle()
                state.update(status='running', current_operator=op)
                m.save(state_path, state)
                suffix = queue.batch.name+'-paired-hgkda'
                run_root = queue.runtime/'kg-runs'
                workspace = run_root/(op+'-'+suffix)
                output = queue.batch/'results'/op/'paired-HG-KDA'
                output.mkdir(parents=True, exist_ok=True)
                # Codex先验和修复明确作为seed来源记录，不称纯模型发现。
                result = {} if config.seed_plan else json.loads(candidate_path.read_text())['results'].get(op, {})
                seed = (queue.batch/'candidates'/(op+'-rawptr-v1.py')
                        if result.get('status') == 'passed' else queue.batch/queue.records[op]['seed'])
                m.save(output/'input-provenance.json', dict(seed=str(seed),
                    seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                    paired_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    attribution='本批plan冻结源码为起点；Codex诊断与整合另记，本组记录CC追加生成'
                                if config.seed_plan else '既有KG候选＋Codex成对诊断修复为起点；本组记录CC追加生成'))
                exports = dict(KG_SOURCE_ROOT=str(overlay), KG_RUN_ROOT=str(run_root),
                    KG_CATALOG_PATH=str(queue.batch/'catalog'), KG_WALLTIME_PORT='19659',
                    KG_PROFILE_PORT='19659', KG_MODEL='deepseek-flash')
                body = '\n'.join('export '+key+'='+shlex.quote(value) for key,value in exports.items())
                body += '\n'+quote(['bash', queue.project/'tools/evaluation/run-kg-operator-910b.sh',
                    op, 'profile', suffix, str(config.rounds), seed])
                if workspace.exists():
                    process = workspace/'.kernelgen/run-process.json'
                    if not config.resume or not process.exists() or not json.loads(process.read_text()).get('finished_at'):
                        raise RuntimeError('已有worker尚未确认退出，不重复启动')
                else:
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
                    if dest.exists():
                        marker = dest/('provenance.json' if script.startswith('export') else 'archive-manifest.json')
                        if not config.resume or not marker.is_file():
                            raise RuntimeError('已有导出缺少完成标记，不覆盖：'+str(dest))
                        continue
                    command = [queue.py, queue.project/'tools/evaluation'/script, workspace]
                    if script.startswith('export'):
                        command.append(op)
                    command.append(dest)
                    queue.run(script+'-paired-'+op, quote(command), 600)
                source = output/'export/main.py'
                semantic_path = output/'semantics.json'
                if not semantic_path.exists():
                    try:
                        queue.run('paired-kg-semantics-'+op, quote([queue.py, tools/'check-pointwise-semantics.py',
                            '--operator', op, '--source', source, '--output', semantic_path]), 1800)
                    except RuntimeError:
                        if not semantic_path.exists():
                            raise
                semantic = json.loads(semantic_path.read_text())
                if semantic['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise RuntimeError('恢复时语义检查与当前源码不同')
                if semantic['status'] != 'passed':
                    state['results'][op] = dict(status='semantic_failed', source=str(source),
                        passed=semantic['passed'], total=semantic['total'])
                    m.save(state_path, state)
                    continue
                if config.dense:
                    dense_path = output/'dense-semantics.json'
                    if not dense_path.exists():
                        try:
                            queue.run('paired-kg-dense-'+op, quote([queue.py, tools/'check-dense-pointwise.py',
                                '--operator', op, '--source', source, '--output', dense_path]), 1800)
                        except RuntimeError:
                            if not dense_path.exists():
                                raise
                    dense = json.loads(dense_path.read_text())
                    if dense['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
                        raise RuntimeError('恢复时连续检查与当前源码不同')
                    if dense['status'] != 'passed':
                        state['results'][op] = dict(status='dense_semantic_failed', source=str(source),
                            passed=dense['passed'], total=dense['total'])
                        m.save(state_path, state)
                        continue
                formal = output/'formal'
                completed = formal/'paired-HG-KDA-1.result.json'
                if config.resume and completed.exists():
                    saved = json.loads(completed.read_text())
                    request = json.loads((formal/'paired-HG-KDA-1.request.json').read_text())
                    if (saved['status'] != 'PASSED' or saved['num_passed'] != saved['num_workloads']
                            or request['implementation']['sources'][0]['content'] != source.read_text()):
                        raise RuntimeError('已有正式结果与源码不匹配，不重复提交')
                    state['results'][op] = dict(status='passed', scores=[saved['geo_mean']],
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        workloads=saved['num_workloads'], output=str(formal))
                else:
                    state['results'][op] = queue.evaluate(op, source, formal, 'paired-HG-KDA', 1)
                m.save(state_path, state)
            state.update(status='completed', current_operator=None)
        except Exception as error:
            state.update(status='needs_attention', error=str(error))
            raise
        finally:
            m.save(state_path, state)


if __name__ == '__main__':
    main()
