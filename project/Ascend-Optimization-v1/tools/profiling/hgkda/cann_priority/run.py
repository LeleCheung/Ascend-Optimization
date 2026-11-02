"""CANN优先的可恢复tmux队列；每项资格和优化结果单独记录。"""
import argparse
import ast
import fcntl
import shlex
from pathlib import Path
from common import BATCH, HERE, NEXT, PY, ROOT, THREE, load, read, save, sha

base = load('cann_priority_next', NEXT / 'run.py')


def quote(parts):
    return ' '.join(shlex.quote(str(p)) for p in parts)


class Runner(base.Runner):
    def supplementary(self, op, source, gates, label):
        if op in {'_conj_physical','expm1_'}:
            leaf = self.batch/self.q.records[op]['catalog_leaf']
            return self.supplement_at(op, source, gates, label, leaf)['status']
        return super().supplementary(op, source, gates, label)

    def save(self):
        save(self.path, self.state)
        lines = ['# CANN优先对照进展', '', '|任务|状态|结果|', '|---|---|---|']
        for key, value in self.state.get('jobs', {}).items():
            result = value.get('result') or {}
            detail = {k: result[k] for k in ['status', 'geo_mean', 'num_passed', 'num_workloads', 'passed', 'total'] if k in result}
            if value.get('error'):detail['error'] = value['error'][:400]
            lines.append('|' + key + '|' + value['status'] + '|' + str(detail).replace('|', '/') + '|')
        (self.batch / '进展.md').write_text(chr(10).join(lines) + chr(10), encoding='utf8')

    def cmd(self, label, args, budget=2100):
        self.q.run(label, quote(args), budget)

    def formal(self, op, source, output, label, leaf=None, kind='CANN_reference'):
        leaf = leaf or self.batch / self.q.records[op]['catalog_leaf']
        self.cmd(label + '-' + op, [PY, HERE / 'evaluate.py', '--operator', op,
                 '--source', source, '--leaf', leaf, '--output', output, '--server', self.q.server,
                 '--label', label, '--kind', kind])
        path = output / (label + '-1.result.json')
        value = read(path)
        return dict(status=value.get('status'), geo_mean=value.get('geo_mean'), num_passed=value.get('num_passed'),
                    num_workloads=value.get('num_workloads'), source=str(source), source_sha256=sha(source),
                    result=str(path), kind=kind, is_hack=value.get('is_hack'))

    def alternate(self, op, reference):
        old = self.batch / self.q.records[op]['catalog_leaf']
        leaf = self.batch / 'cann-catalog' / sha(reference)[:12] / 'ops' / op
        leaf.mkdir(parents=True, exist_ok=True)
        for name in ['definition.json', 'correctness.jsonl', 'timing.jsonl']:
            data = (old / name).read_bytes()
            p = leaf / name
            if p.exists():assert p.read_bytes() == data
            else:p.write_bytes(data)
        text = (old / 'oracle.py').read_text(encoding='utf8')
        # 保留唯一设备声明与原gen_inputs/valid；只覆盖目标run及其CANN桥。
        text += '\n\n# 本批独立CANN参考；原合同保留在catalog。\n' + reference.read_text(encoding='utf8')
        tree = ast.parse(text)
        original = ast.parse((old / 'oracle.py').read_text(encoding='utf8'))
        gen = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'gen_inputs')
        assert ast.dump(gen(tree)) == ast.dump(gen(original))
        p = leaf / 'oracle.py'
        if p.exists():assert p.read_text(encoding='utf8') == text
        else:p.write_text(text, encoding='utf8')
        save(leaf.parents[1] / 'manifest.json', dict(api_version='v6.2', name=self.batch.name + '-CANN',
              evaluator='native', layout='per-operator'))
        save(self.batch / 'comparison-provenance' / (op + '.json'), dict(
             reference=str(reference), reference_sha256=sha(reference), original_contract_sha256=self.q.records[op]['contract_sha256'],
             new_oracle_sha256=sha(p), gen_inputs_ast_identical=True, scope='只替换命名性能参考；原PyTorch资格另行保存'))
        return leaf

    def reference(self, op, item):
        source = Path(item['source'])
        assert sha(source) == item['source_sha256']
        out = source.parent
        version = ':' + sha(source)[:12] + ':eval-' + sha(HERE/'evaluate.py')[:8]
        formal = self.job('reference-formal:' + op + version, lambda: self.formal(op, source, out / 'formal', 'CANN', kind=item['kind']))
        entry = dict(item, formal=formal)
        if formal and formal['status'] == 'PASSED' and not formal.get('is_hack'):
            gates = out / 'extra-checks.json'
            checker_version = ''
            if op == 'expm1_':
                checker_version = ':checker-' + sha(HERE/'validate_cann.py')[:12]
                gates = gates.with_name('extra-checks-' + sha(HERE/'validate_cann.py')[:12] + '.json')
            qualification = self.job('reference-extra:' + op + version + checker_version, lambda: dict(status=self.supplementary(op, source, gates, 'CANN-extra-' + op)))
            entry.update(supplementary=qualification, extra_checks=str(gates))
            if qualification and qualification['status'] == 'passed':
                leaf = self.alternate(op, source)
                candidate = self.starting_source(op)
                comparison = self.job('candidate-vs-cann:' + op + version, lambda: self.formal(op, candidate,
                    self.batch / 'comparisons' / op / 'starting', 'HG-KDA-start', leaf, 'existing_candidate_vs_CANN'))
                entry['comparison'] = comparison
                if comparison and comparison['status'] == 'PASSED':
                    gates = self.batch / 'comparisons' / op / 'starting/extra-vs-cann.json'
                    checker_version = ''
                    if op in {'hardshrink_backward','expm1_'}:
                        checker_version = ':checker-' + sha(HERE/'validate_cann.py')[:12]
                        gates = gates.with_name('extra-vs-cann-' + sha(HERE/'validate_cann.py')[:12] + '.json')
                    entry['comparison_supplementary'] = self.job('candidate-extra-vs-cann:' + op + version + checker_version,
                        lambda: self.supplement_at(op, candidate, gates, 'starting-extra-vs-cann-' + op, leaf))
        self.state.setdefault('references', {})[op] = entry
        self.save()
        return entry

    def supplement_at(self, op, source, gates, label, leaf):
        from validation_status import supplement_status
        if not gates.exists():
            checker = (HERE.parent/'reference_closure/validate_reference.py' if op == '_conj_physical' else
                       HERE/'validate_cann.py' if op in {'hardshrink_backward','expm1_'} else NEXT/'validate.py')
            try:
                self.cmd(label, [PY, checker, '--operator', op, '--source', source,
                         '--leaf', leaf, '--output', gates], 1200)
            except RuntimeError:
                if not gates.exists():raise
        value = read(gates)
        return dict(status=supplement_status(op, value, sha(source)), source_sha256=sha(source),
                    extra_checks=str(gates), passed=value.get('passed'), total=value.get('total'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'references', 'optimize', 'priority', 'search', 'extra'], default='all')
    args = parser.parse_args()
    with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = Runner(BATCH.name)
        if args.phase in {'all', 'references', 'priority'}:
            build_key = 'build-cann:' + sha(HERE / 'cann_baselines.cpp')[:12]
            r.job(build_key, lambda: r.cmd('build-cann', [PY, HERE / 'build.py'], 1200))
            if not (BATCH / 'cann-build-v1.json').exists():
                raise RuntimeError('CANN参考构建未完成')
            r.job('make-references', lambda: r.cmd('make-references', [PY, HERE / 'make_references.py'], 180))
            refs = read(BATCH / 'reference-index.json')
            priority = ['hardswish_out', 'hardswish', 'selu_', 'native_group_norm', 'negative_', 'log2_', 'nan_to_num_']
            order = ['hardswish_out'] if args.phase == 'priority' else priority + [n for n in refs if n not in priority]
            for op in order:
                r.reference(op, refs[op])
                if op == 'hardswish_out' and args.phase in {'all', 'priority'}:
                    optimize(r)
        elif args.phase == 'extra':
            key = 'build-storage-rng:' + sha(HERE/'cann_storage_rng.cpp')[:12]
            r.job(key, lambda: r.cmd('build-storage-rng', [PY, HERE/'build_storage_rng.py'], 1200))
            if not (BATCH/'cann-build-storage-rng-v1.json').exists():
                raise RuntimeError('存储与随机参考构建失败，保留错误原件')
            index_version=sha(HERE/'repair_index.py')[:12]
            upsample_build=BATCH/'upsample-reference-build-v2.json'
            if upsample_build.exists():index_version+=':'+sha(upsample_build)[:12]
            r.job('reference-index-storage:'+index_version, lambda: r.cmd('reference-index-storage',[PY,HERE/'repair_index.py'],180))
            refs = read(BATCH/'reference-index.json')
            for op in ['uniform','_conj_physical','hardshrink_backward','expm1_','upsample_bilinear2d']:
                r.reference(op,refs[op])
            optimize(r)
        elif args.phase == 'search':
            from search_missing import process
            process(r)
            optimize(r)
        else:
            optimize(r)
        if args.phase == 'all':
            optimize(r)
        r.wait_idle()
        r.state.update(status='phase_completed', current_job=None, finished_at=r.module.now())
        r.save()
        r.cmd('final-report', [PY, HERE/'report.py'], 180)


def optimize(r):
    for proposal in sorted((BATCH / 'proposals').glob('*/*/proposal.json')):
        item = read(proposal)
        if item.get('disabled_reason'):
            continue
        op = item['operator']
        source = proposal.parent / item['source']
        assert sha(source) == item['source_sha256']
        out = proposal.parent
        if op in {'hardswish','hardswish_out','rad2deg_'} and item['backend'].startswith('AscendC'):
            native = out/'native-checks.json'
            def native_gate(op=op,source=source,out=out,native=native):
                if not native.exists():
                    try:
                        r.cmd('native-hardswish-'+op+'-'+out.name,[PY,HERE/'check_hardswish_native.py',
                            '--operator',op,'--source',source,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],'--output',native],1200)
                    except RuntimeError:
                        if not native.exists():raise
                value=read(native)
                assert value['source_sha256']==sha(source)
                return dict(status=value['status'],passed=value['passed'],total=value['total'])
            checked=r.job('proposal-native:'+op+':'+out.name,native_gate)
            if not checked or checked['status']!='passed':continue
        gates = out / 'extra-checks.json'
        qualification = r.job('proposal-extra:' + op + ':' + out.name,
            lambda op=op, source=source, gates=gates, out=out: dict(status=r.supplementary(op, source, gates, 'proposal-extra-' + op + '-' + out.name)))
        if not qualification or qualification['status'] != 'passed':
            continue
        result = r.job('proposal-original:' + op + ':' + out.name,
            lambda op=op, source=source, out=out: r.formal(op, source, out / 'original', 'HG-KDA-new', kind=item['backend']))
        ref = r.state.get('references', {}).get(op)
        if result and result['status'] == 'PASSED' and ref and ref.get('supplementary', {}).get('status') == 'passed':
            leaf = r.alternate(op, Path(ref['source']))
            compared = r.job('proposal-vs-cann:' + op + ':' + out.name,
                lambda op=op, source=source, out=out, leaf=leaf: r.formal(op, source, out / 'CANN', 'HG-KDA-new-vs-CANN', leaf, item['backend']))
            if compared and compared['status'] == 'PASSED':
                extra = r.job('proposal-extra-vs-cann:' + op + ':' + out.name,
                    lambda op=op, source=source, out=out, leaf=leaf: r.supplement_at(op, source,
                        out/'CANN/extra-checks.json', 'new-extra-vs-cann-'+op+'-'+out.name, leaf))
                if extra and extra['status'] == 'passed' and compared.get('geo_mean', 0) > .8:
                    from capture import capture
                    r.job('proposal-paired-paths:' + op + ':' + out.name,
                        lambda op=op, source=source, leaf=leaf, out=out, ref=ref: capture(r, op, source, Path(ref['source']), leaf, out.name))


if __name__ == '__main__':
    main()
