"""沿用已验证KG引擎，给新工作区注入最新三类型指标和编译机制证据。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'
ENGINE = HERE.parent / 'strict_night'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def run(attempt):
    sys.path.insert(0, str(ENGINE))
    spec = importlib.util.spec_from_file_location('strict_next_search_engine', ENGINE / 'run.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.BATCH = BATCH

    class Search(base.Runner):
        def __init__(self):
            super().__init__()
            self.path = BATCH / 'search-state.json'
            self.state = read(self.path) if self.path.exists() else dict(status='created', jobs={}, operators={}, started_at=self.module.now())
            self.q.plan['profile_budget_seconds'] = 240
            domain = BATCH / 'analysis/expm1_/bfloat-domain-frozen-best-v1/result.json'
            if domain.exists() and read(domain)['status'] == 'failed':
                old = self.previous['expm1_']
                assert read(domain)['source_sha256'] == old['source_sha256']
                old['supplementary'] = dict(status='failed', original=old.get('supplementary'),
                    new_evidence=str(domain), scope='仅本轮候选选择门槛采用新增BF16实测，不改写冻结delivery或历史结果')

        def process_proposals(self):
            from measurements import baseline_score
            for label, domain in [
                    ('single-buffer-layout-v2', BATCH / 'analysis/expm1_/bfloat-domain-buffer-v2-v1/result.json'),
                    ('bf16-wide-v3', BATCH / 'compile-probes/bf16-wide-v3/full-bf16-domain/result.json')]:
                folder = BATCH / 'compile-probes' / label
                result_path = folder / 'result.json'
                if not result_path.exists() or not domain.exists():
                    continue
                result, checked = read(result_path), read(domain)
                if not result.get('qualified_original') or checked['status'] != 'passed':
                    continue
                assert checked['source_sha256'] == result['source_sha256'] == sha(folder / 'main.py')
                if result['PyTorch'] <= baseline_score(self.previous['expm1_']):
                    continue
                candidate = folder / 'verified-candidate.json'
                if candidate.exists():
                    best = read(candidate)
                else:
                    review = read(folder / 'review.json')
                    assert review['source_sha256'] == result['source_sha256'] and review['all_branches_reviewed']
                    best = dict(source=str(folder / 'main.py'), source_sha256=result['source_sha256'],
                        search_result=result['formal'], search_score=result['PyTorch'], qualification='passed',
                        review_status='reviewed_strict_triton', reference='original_PyTorch_only',
                        pytorch_result=result['formal'], kind='Codex_proposal',
                        provenance=read(folder / 'spec.json')['attribution'],
                        extras={'pytorch': dict(path=result['supplement'], status='passed')})
                    final_spec = importlib.util.spec_from_file_location('strict_next_independent_finalize', ENGINE / 'finalize.py')
                    final = importlib.util.module_from_spec(final_spec)
                    final_spec.loader.exec_module(final)
                    final.BATCH = BATCH
                    final.verify(self, 'expm1_', best)
                    save(candidate, best)
                row = self.state['operators'].setdefault('expm1_', dict(attempts=[]))
                proposals = row.setdefault('proposals', [])
                if not any(r['source_sha256'] == best['source_sha256'] for r in proposals):
                    best['bfloat_domain'] = str(domain)
                    proposals.append(best)
                    self.refresh_existing_scores()

        def verify_reviewed_best(self):
            pass

        def prepare_paired(self, op, seed):
            pass

        def validate(self, op, source, out, existing_pytorch_result=None):
            value = super().validate(op, source, out, existing_pytorch_result)
            if value['qualification'] != 'passed':
                return value
            exact_tool = HERE.parent / 'strict_night/expm1_exact_probe.py'
            exact_frozen = out / ('exact-checker-' + sha(exact_tool) + '.py')
            if exact_frozen.exists():
                assert exact_frozen.read_bytes() == exact_tool.read_bytes()
            else:
                exact_frozen.write_bytes(exact_tool.read_bytes())
            exact = out / 'exact-expm1-probe'
            if not (exact / 'result.json').exists():
                self.cmd('exact-' + sha(source)[:12], [base.PY, exact_frozen, '--source', source, '--output', exact], 900)
            from expm1_exact_probe import qualified
            value['exact_probe'] = str(exact / 'result.json')
            if not qualified(read(exact / 'result.json'), sha(source)):
                value.update(qualification='failed', compiled_status='not_qualified')
                return value
            tool = HERE.parent / 'strict_five/probe_half_domain.py'
            frozen = out / ('domain-checker-' + sha(tool) + '.py')
            if frozen.exists():
                assert frozen.read_bytes() == tool.read_bytes()
            else:
                frozen.write_bytes(tool.read_bytes())
            value['full_16bit_domains'] = {}
            for dtype in ['float16', 'bfloat16']:
                output = out / ('full-' + dtype + '-domain')
                path = output / 'result.json'
                if not path.exists():
                    self.cmd('domain-' + dtype + '-' + sha(source)[:12],
                             [base.PY, frozen, '--source', source, '--output', output, '--dtype', dtype], 1200)
                checked = read(path)
                assert checked['source_sha256'] == sha(source) and checked['checker_sha256'] == sha(frozen)
                value['full_16bit_domains'][dtype] = dict(path=str(path), status=checked['status'])
                if checked['status'] != 'passed':
                    value.update(qualification='failed', compiled_status='not_qualified')
            return value

        def report(self):
            save(BATCH / 'search-progress.json', dict(status=self.state['status'], current_job=self.state.get('current_job'),
                best={op: {key: (row.get('best') or {}).get(key) for key in ['source', 'source_sha256', 'search_score', 'qualification']}
                      for op, row in self.state['operators'].items()}, frozen_accepted=25,
                scope='KG搜索结果仍需人工审查、全部补测及实际后端核验；报告或搜索结束不等于30项达标'))

        def context(self, op, seed):
            previous = self.previous[op]
            formal = read(previous['pytorch']['path'])
            paired = []
            metrics = {'aiv_time(us)', 'aiv_vec_time(us)', 'aiv_scalar_time(us)', 'aiv_mte2_time(us)', 'aiv_mte3_time(us)',
                       'aiv_scalar_wait_time(us)', 'aiv_mte2_instructions', 'aiv_mte3_instructions',
                       'GM_to_UB_datas(KB)', 'UB_to_GM_datas(KB)', 'read_main_memory_datas(KB)', 'write_main_memory_datas(KB)'}
            for dtype in ['float16', 'bfloat16', 'float32']:
                path = BATCH / 'analysis/expm1_' / ('paired-best-' + dtype) / 'summary.json'
                value = read(path)
                assert value['status'] == 'completed' and value['spec']['source_sha256'] == previous['source_sha256']
                data = value.get('comparison') or {}
                selected = [v for v in data.get('differences', []) if (
                    v['table'] in ['PipeUtilization.csv/vector0', 'Memory.csv/vector0'] and v['metric'] in metrics)
                    or (v['table'] == 'ResourceConflictRatio.csv/vector0' and 'conflict' in v['metric'].lower())]
                paired.append(dict(dtype=dtype, summary=str(path), sha256=sha(path), spec=value['spec'],
                                   differences=selected, derived_differences=data.get('derived_differences'),
                                   actual_ir=str(path.parent / 'candidate/ir/manifest.json')))
            proposals = []
            proposal_labels = dict(zip(['codex-fma-series-v1', 'codex-typed-fma-v2', 'single-buffer-layout-v2', 'bf16-wide-v3'],
                                      ['compile-fma-v1', 'compile-typed-fma-v2', 'compile-single-buffer-v2', 'compile-bf16-wide-v3']))
            folders = list((BATCH / 'proposals/expm1_').iterdir())
            folders += [BATCH / 'compile-probes' / name for name in ['single-buffer-layout-v2', 'bf16-wide-v3']]
            for folder in sorted(folders):
                if not folder.exists():
                    continue
                item = dict(proposal=folder.name, source=str(folder / 'main.py'), source_sha256=sha(folder / 'main.py'))
                result = folder / 'result.json'
                if result.exists():
                    item['result'] = read(result)
                compile_name = proposal_labels[folder.name]
                path = BATCH / 'analysis/expm1_' / compile_name / 'result.json'
                if path.exists():
                    item['compilation'] = dict(path=str(path), status=read(path)['status'], rows=[
                        {key: v.get(key) for key in ['dtype', 'symbol', 'block', 'status', 'ttir_fma_count']} for v in read(path)['rows']])
                proposals.append(item)
            history = []
            for attempt in self.state['operators'].get(op, {}).get('attempts', []):
                history.append(dict(attempt=attempt['attempt'], workspace=attempt['workspace'],
                    status=attempt['status'], candidates=[{k: c.get(k) for k in
                        ['source', 'source_sha256', 'search_score', 'qualification', 'manual_review_rejection',
                         'extras', 'full_16bit_domains']} for c in attempt.get('candidates', [])]))
            rejections = [read(path) for path in sorted((BATCH / 'review-rejections' / op).glob('*.json'))]
            domains = []
            for path in sorted((BATCH / 'analysis/expm1_').glob('bfloat-domain-*/result.json')):
                checked = read(path)
                domains.append(dict(path=str(path), source_sha256=checked['source_sha256'], status=checked['status'],
                    cases=[dict(case=c['case'], passed=c['passed'], matched=c.get('matched'), total=c.get('total'),
                                counterexamples=c.get('counterexamples', [])[:4]) for c in checked['cases']]))
            selected = self.state['operators'].get(op, {}).get('best') or {}
            return dict(operator=op, current_seed=dict(path=str(seed), sha256=sha(seed)),
                frozen_score=previous['pytorch']['score'], frozen_source_sha256=previous['source_sha256'],
                per_case=[{k: r.get(k) for k in ['uuid', 'phase', 'latency_ms', 'reference_latency_ms']}
                          for r in formal['per_workload'] if r['phase'] == 'timing'], paired_hardware_evidence=paired,
                independently_tested_proposals=proposals,
                previous_searches=history, manual_review_rejections=rejections,
                full_bfloat16_domain_evidence=domains, selected_best=selected,
                strict_contract='所有设备计算和回退必须自包含Python+Triton；不使用CANN/Ascend C/PyTorch计算/原生DMA或外部.so。原地每次只执行一次；只可缓存编译及元数据，不缓存输入、输出、旧stream或结果。',
                target='先实质改善当前0.646557×的expm1完整调用，争取超过0.8×；CANN参考资格仍单列，不用不准确参考凑分。',
                diagnosis='新最佳三类型总GM读写与参考相同，Vector时间却分别约2.06/2.96/4.46倍；MTE时间未同比恶化。优先核对数学运算、类型扩展、UB存活、流水与同步，而不是直接归咎HBM带宽。',
                compiler_observation='显式tl.fma(x,0.5,1.0)的默认常量在TTIR将FP16扩展为FP32，使FP16流式需求1835264 bits超过1572864、BF16也溢出；同dtype常量提案及实际结果在independently_tested_proposals。不能把更少源代码运算等同于更少设备指令。',
                search_directions=[
                    '读取实际TTIR和adapter/后续IR，看Vector指令和缓冲是否真正减少，结合已成功或失败的同dtype常量提案。不要重新完整评测同一代码。',
                    'BF16会升为FP32执行部分运算，和FP16分别选择组织方式；检查混合精度、块内复用、同步、每核循环及软件流水。',
                    '只在新的机制证据支持时试编译器num_stages、multibuffer、enable_ubuf_saving等正常选项，实际UB编译和数值先过，不能关闭必要同步或换后端。',
                    '保留小值相对精度、负零、NaN/Inf、stride、尾部、非对齐、当前stream及冷启动只执行一次；不能直接删掉稳定公式以追赶不准确的参考。'],
                previous_failures='上批纯FP16数学0.542×，仅消除比较扩展0.486×，都不优于0.647×；不要重复旧分块和同源码挑分。各dtype两块/单块的旧数值只是历史事实，不能写进新稿注释充当本轮测量。',
                new_requirements='遵循manual_review_rejections的实际缓存约束；未对齐地址不能复用仅对齐地址warmup的实例。缓存保留设备和地址特化身份；可按实际地址JIT回退，不缓存输入/输出/stream。新候选补测全部FP16及BF16位模式四类实际路径；若BF16新证据失败，先修复约0.125边界附近的消减误差，再继续提速。保持完整原合同及全部特殊值。',
                workflow='KG/HG-KDA与CC+DeepSeek生成。Codex提供新配对指标、IR机制判断及独立提案；不是纯原生KG独立效果。先给出可运行的代码改进并完整evaluate，单次debug<=120秒，沿用现有诊断预算。')

    method = Path(__file__)
    frozen = BATCH / 'method-snapshots' / sha(method) / method.name
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists():
        assert frozen.read_bytes() == method.read_bytes()
    else:
        frozen.write_bytes(method.read_bytes())
    save(BATCH / ('search-loaded-methods-' + str(attempt) + '.json'),
         dict(search_sha256=sha(method), frozen=str(frozen), engine_sha256=sha(ENGINE / 'run.py'),
              domain_checker_sha256=sha(HERE.parent / 'strict_five/probe_half_domain.py')))
    runner = Search()
    assert runner.state.get('current_job') is None, '未知在途搜索不能重发'
    runner.refresh_existing_scores()
    cache = BATCH / 'evaluation-cache'
    cache.mkdir(exist_ok=True)
    for name in ['strict-night-20261102', 'strict-five-20261103']:
        old = BATCH.parent / name / 'evaluation-cache'
        for source in old.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            target = cache / source.relative_to(old)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == source.read_bytes(), '缓存冲突不能选更快成绩'
            else:
                shutil.copyfile(source, target)
    runner.search('expm1_', attempt)
    runner.state.update(status='search_completed_pending_review', current_job=None, finished_at=runner.module.now())
    runner.persist()


if __name__ == '__main__':
    run(int(sys.argv[1]))
