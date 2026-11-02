"""严格Triton持续优化队列：固定CANN对照，持卡锁，已知终态才交接。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
OLD = PROJECT/'operators/campaigns/strict-triton-20261102'
BATCH = PROJECT/'operators/campaigns/strict-night-20261102'
PY = '/usr/local/python3.11.15/bin/python3.11'
PRIORITY = ['hardswish', 'fix', 'hardshrink_backward', 'rad2deg_', 'round_out',
            'special_round_out', 'hardswish_out', 'log2_', 'fmod_tensor_']
REMAINDER = ['cumulative_trapezoid', 'linalg_inv_ex', 'expm1_', 'uniform', 'fractional_max_pool3d']


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf8'); tmp.replace(p)
def module(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def quote(args): return ' '.join(shlex.quote(str(v)) for v in args)


def prepare():
    if (BATCH/'plan.json').exists(): return
    BATCH.mkdir(parents=True, exist_ok=True)
    previous = read(OLD/'delivery.json')
    assert previous['accepted'] == 16
    save(BATCH/'starting-delivery.json', previous)
    plan = read(OLD/'plan.json')
    rows = {r['operator']: r for r in previous['rows']}
    for record in plan['records']:
        op = record['operator']; row = rows[op]
        seed = BATCH/'starting-points'/(op+'.py'); seed.parent.mkdir(exist_ok=True)
        source = Path(row['source']); assert sha(source) == row['source_sha256']
        shutil.copyfile(source, seed)
        for name, origin in [('original-catalog', OLD/'catalog/ops'/op),
                             ('catalog', OLD/('cann-catalog' if row.get('cann') else 'catalog')/'ops'/op)]:
            dest = BATCH/name/'ops'/op
            shutil.copytree(origin, dest, ignore=shutil.ignore_patterns('__pycache__'))
        record.update(seed=str(seed.relative_to(BATCH)), source_sha256=sha(seed),
                      catalog_leaf='catalog/ops/'+op,
                      contract_sha256={n:sha(BATCH/'catalog/ops'/op/n)
                                       for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
                      reference='qualified_CANN' if row.get('cann') else 'original_PyTorch_only')
    for name in ['catalog', 'original-catalog']:
        save(BATCH/name/'manifest.json', dict(api_version='v6.2', name=BATCH.name+'-'+name,
                                            evaluator='native', layout='per-operator'))
    plan.update(schema='hgkda-strict-night-v1', priority=PRIORITY, remainder=REMAINDER,
                duration_hours=18, passes=[dict(rounds=4,budget_seconds=2400)],
                starting_delivery_sha256=sha(OLD/'delivery.json'),
                baseline_commit='a4cb1b30f9b93d600a0c50d698af1f14d826bb39',
                candidate_backend='strict_Triton', reference_policy='合格原生CANN；缺口仅按原语义搜索，不计CANN达标',
                measurement_policy='每份新源码默认一次完整评测；原合同不变；禁止同SHA重复挑分')
    save(BATCH/'plan.json', plan)


class Runner:
    def __init__(self):
        self.module = module('night_queue', TOOLS/'run-nightly.py')
        self.module.save = save
        self.q = self.module.Queue(BATCH.name, ROOT, 'tle_yy')
        self.batch = BATCH
        self.path = BATCH/'state.json'
        self.state = read(self.path) if self.path.exists() else dict(status='running', jobs={}, operators={}, started_at=self.module.now())
        self.previous = {r['operator']: r for r in read(BATCH/'starting-delivery.json')['rows']}
        sys.path.insert(0, str(TOOLS/'next_phase'))
        self.native_method = module('night_native', TOOLS/'next_phase/native_search.py').native
    def persist(self): save(self.path, self.state); self.report()
    def process_proposals(self):
        module('strict_night_proposals',HERE/'proposals.py').process(self)
    def verify_reviewed_best(self):
        helper=None
        for op,row in self.state['operators'].items():
            best=row.get('best') or {}
            if best.get('qualification')!='passed':continue
            review=Path(best['source']).with_name('review.json')
            if not review.exists():continue
            diagnostic_pending=(op=='fix' and read(review).get('diagnostic_parent_source_sha256')
                                and not best.get('component_diagnostic'))
            if best.get('compiled_status')=='verified' and not diagnostic_pending:continue
            if helper is None:helper=module('night_backend_finalize',HERE/'finalize.py')
            helper.verify(self,op,best)
            save(self.path,self.state)
    def manual_rejection(self,op,source):
        path=BATCH/'review-rejections'/op/(sha(source)+'.json')
        if not path.exists():return None
        value=read(path)
        assert value['operator']==op and value['source_sha256']==sha(source)
        assert value['status']=='manual_review_rejected' and value.get('reason')
        return str(path)
    def refresh_existing_scores(self):
        from measurements import ledger,score,baseline_score
        for op,row in self.state['operators'].items():
            qualified=[]
            for attempt in row.get('attempts',[]):
                for entry in attempt.get('candidates',[]):
                    result=Path(entry['search_result']);source=Path(entry['source'])
                    workspace=next(p for p in result.parents if (p/'.kernelgen/run-process.json').exists())
                    contract=read(workspace/'stages/optimize/work/.kernelgen/evaluation-contract.json')
                    expected=(BATCH/'catalog/ops'/op/'oracle.py').read_text()
                    assert contract['definition']['reference']==expected,'实际KG合同不是冻结参考'
                    value=ledger(workspace,source);save(source.parent/'existing-measurements.json',value)
                    entry.setdefault('search_round_score',entry['search_score'])
                    entry['search_score']=value['score'];entry['measurement_ledger']=str(source.parent/'existing-measurements.json')
                    if value['status']!='passed':entry['qualification']=value['status']
                    rejection=self.manual_rejection(op,source)
                    if rejection:entry.update(qualification='manual_review_rejected',manual_review_rejection=rejection,compiled_status='not_qualified')
                    if entry.get('qualification')=='passed':qualified.append(entry)
            for entry in row.get('proposals',[]):
                assert sha(entry['source'])==entry['source_sha256']
                value=read(entry['search_result'])
                try:entry['search_score']=score(value)
                except (AssertionError,KeyError,ValueError,TypeError):entry['qualification']='formal_failure'
                if entry.get('qualification')=='passed':qualified.append(entry)
            if qualified:
                best=max(qualified,key=lambda x:x['search_score'])
                baseline=baseline_score(self.previous[op])
                if best['search_score']>baseline:row['best']=best
                else:row.pop('best',None)
            else:row.pop('best',None)
    def recover_meaningful_selection(self):
        """旧控制器可能被排版变化高分截断；只补验已结束工作区里漏选的实质候选。"""
        from measurements import computation_identity
        for op,row in self.state['operators'].items():
            if row.get('best'):continue
            baseline_id=computation_identity(BATCH/self.q.records[op]['seed'])
            for attempt in row.get('attempts',[]):
                workspace=Path(attempt['workspace'])
                metadata=read(workspace/'.kernelgen/run-process.json')
                assert metadata.get('finished_at'),'搜索尚未结束，不恢复选择'
                seen={r['source_sha256'] for a in row['attempts'] for r in a.get('candidates',[])}
                missing=[]
                for p in workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/main.py'):
                    result=p.with_name('result.json')
                    if sha(p) in seen or not result.exists() or read(result).get('status')!='PASSED':continue
                    try:changed=computation_identity(p)!=baseline_id
                    except SyntaxError:continue
                    if changed:missing.append(p)
                if not missing:continue
                recovered=self.collect(op,attempt['attempt'],workspace)
                attempt.setdefault('selection_recoveries',[]).append(recovered)
                attempt['candidates'].extend(recovered['candidates'])
        self.refresh_existing_scores()
    def wait(self):
        end = time.monotonic()+900
        while True:
            try: self.q.idle(); return
            except self.module.Attention:
                if time.monotonic()>end: raise
                time.sleep(10)
    def cmd(self,key,args,budget=2100):
        self.wait(); return self.q.run(key, quote(args), budget)
    def context(self,op,seed):
        old = self.previous[op]
        evidence = dict(operator=op, seed=dict(path=str(seed),sha256=sha(seed)),
                        starting_CANN=old.get('cann'), starting_PyTorch=old.get('pytorch'),
                        device_path=old.get('device_path'), compiled_path=old.get('compiled_path'))
        formal = old.get('cann') or old.get('pytorch')
        if formal:
            v = read(formal['path'])
            evidence['per_case'] = [{k:r.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms']}
                                    for r in v.get('per_workload',[]) if r.get('phase')=='timing']
        selected=(self.state.get('operators',{}).get(op) or {}).get('best') or {}
        if selected:
            evidence['same_operator_measured_best']={k:selected.get(k) for k in
                ['source','source_sha256','search_result','search_score','pytorch_result','qualification','host_glue']}
            evidence['same_operator_measured_best']['scope']='本算子已有独立完整测量；不是冻结起点。人工核验宿主桥不扩张CC自包含Python+Triton准入。'
        if selected.get('source_sha256')==sha(seed) and selected.get('search_result'):
            measured=read(selected['search_result'])
            evidence['selected_seed_measurement']=dict(path=selected['search_result'],
                sha256=sha(selected['search_result']),score=selected['search_score'],
                qualification=selected.get('qualification'),compiled_status=selected.get('compiled_status'),
                per_case=[{k:r.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms']}
                          for r in measured.get('per_workload',[]) if r.get('phase')=='timing'],
                scope='当前seed实际已完成的测量；与冻结起点分开，不重复提交同源码')
        evidence['diagnosis'] = (
            '固定真实CANN参考，目标完整调用几何平均>0.8。检查大输入program数量、单program流式循环与访存组织；'
            '区分宿主发射与设备耗时。分块需控制UB，不得只调小输入延迟。对照源码/实际IR/trace按需读取；'
            '不得把待验证解释写成结论。允许编译缓存和当前stream getter，禁止缓存输出或旧stream。'
            '源码含_strict存储工具时仍须保留非连续分支。先形成一份真实可运行改进并完整evaluate；'
            '首轮前最多两次debug，每次<=120秒。支持普通shape/dtype策略，不按用例名/seed特化。'
            '候选必须完全自包含的Python+Triton；不允许外部.so、CANN/AscendC/PyTorch计算/原生DMA。'
            '当前适合先试减少program重复调度或增大片内顺序处理；用profile验证，别只重测同源码。')
        evidence['bounded_pointwise_knowledge']=dict(
            scope='本环境逐元素类经验；经验阈值仅生成假设，不自动证明瓶颈',
            metric_rules=[
                'MTE或Vector活跃比例>=0.7时检查对应方向；Scalar>=0.5时检查寻址/控制。先看绝对耗时、流量、program记录数和候选/参考范围。',
                '相同大输入FP16/BF16明显慢于参考，检查转换、缓冲存活范围、动态掩码、每核循环与搬运组织，不能仅凭字节数推断HBM受限。'],
            host_rules=[
                '小输入优先减少Python调用层次、shape构造、分配包装和重复metadata查询；缓存编译参数但不能缓存输出、输入值或旧stream。',
                '可将原严格Triton连续化路径直接并入主入口，连续输入用torch.empty_like只分配，非连续输入仍由Triton复制。',
                'fix本批发现新增persistent kernel后raw launcher参数仍沿用旧签名，异常后走JIT。修改形参须同步warmup、raw打包和JIT回退，constexpr也可能占launcher参数位；按当前编译产物核对，不假定跨版本固定ABI。'],
            compiler_rules=[
                '分块放大要同时检查实际UB编译分配。hardswish本批出现需求2097408 bits超过可用1572864 bits，失败原件在admissions中；不能把tile上限一股脑套到所有算子。',
                '完整块与尾部可以分开，但无掩码路径必须由一般边界条件保证；非连续输入、负零、NaN/Inf、参数变化及多stream不能丢失。',
                '原地算子冷启动只warmup编译，随后执行恰好一次。不要先JIT实际计算、再读取launcher并在异常后重新计算；这会在异常路径把数据变换两次。',
                '编译缓存必须区分实际kernel、block、dtype、device及会影响编译的标量类别；raw launcher的constexpr槽不会改变已有二进制block。'])
        rejected=[]
        for path in sorted((BATCH/'review-rejections'/op).glob('*.json')):
            value=read(path)
            rejected.append(dict(path=str(path),sha256=sha(path),source_sha256=value['source_sha256'],reason=value['reason'],repair=value.get('repair')))
        if rejected:evidence['manual_rejected_predecessors']=rejected
        failed_proposals=[]
        for entry in (self.state.get('operators',{}).get(op) or {}).get('proposals',[]):
            if entry.get('qualification')=='passed':continue
            failed={k:entry.get(k) for k in ['source','source_sha256','search_result','search_score','reference','qualification']}
            failed['supplementary_failures']={}
            for side,item in (entry.get('extras') or {}).items():
                if item.get('status')=='passed':continue
                value=read(item['path'])
                failed['supplementary_failures'][side]=dict(path=item['path'],sha256=sha(item['path']),
                    status=value.get('status'),passed=value.get('passed'),total=value.get('total'),
                    cases=[{k:c.get(k) for k in ['case','check','error']} for c in value.get('cases',[])
                           if c.get('passed') is False][:6])
            failed_proposals.append(failed)
        if failed_proposals:evidence['same_operator_failed_proposals']=failed_proposals[-4:]
        transferred=[]
        for prior_op,row in self.state.get('operators',{}).items():
            best=row.get('best') or {}
            if (prior_op==op or best.get('qualification')!='passed'
                    or best.get('compiled_status')!='verified'
                    or best.get('review_status')!='reviewed_strict_triton'):continue
            transferred.append(dict(operator=prior_op,**{k:best.get(k) for k in ['source','source_sha256','search_result','search_score','pytorch_result','review_status','compiled_status','component_diagnostic']}))
        # 优先同类原地/非原地接口，再按本批处理顺序选较后的已核验经验。
        transferred.reverse()
        transferred.sort(key=lambda v:v['operator'].endswith('_')!=op.endswith('_'))
        evidence['same_batch_verified_examples']=transferred[:3]
        if op in {'cumulative_trapezoid','linalg_inv_ex','fractional_max_pool3d'}:
            evidence.pop('bounded_pointwise_knowledge')
            evidence['same_batch_verified_examples']=transferred[:1]
            evidence['example_scope']='逐元素历史仅作宿主启动和验证方法参考，不作为扫描、矩阵求逆或池化的数学/分块模板。'
        paired=BATCH/'analysis'/op/('paired-'+sha(seed)[:16])/'summary.json'
        if paired.exists():
            value=read(paired)
            facts=dict(path=str(paired),sha256=sha(paired),status=value['status'],spec=value['spec'])
            facts['sides']={k:{f:v.get(f) for f in ['status','meaning','kernels','error']} for k,v in value['sides'].items()}
            comparison=value.get('comparison') or {}
            facts['comparison']={k:comparison.get(k) for k in ['path','sha256','derived_differences','limitations']}
            facts['comparison']['differences']=comparison.get('differences',[])[:90]
            evidence['independent_CANN_paired']=facts
            evidence['diagnosis']+=('本环境KGS标准profile曾明确失败：缺少completion marker。'
                '优先使用上述独立msprof op真实证据，不反复重试同样失效的标准采集。'
                '各分量可重叠，不能相加为总耗时；记录均值与完整任务时长区分。'
                '独立采集如失败，明确无指标，只按源码/IR/实测提出假设；不能编造数据。')
        if op in {'round_out','special_round_out'}:
            history=PROJECT/'operators/campaigns/next25-20261029-v2/results/round_out/search-1/native-evidence/files/00030-stdout.log'
            if history.exists():
                evidence['historical_round_intrinsic_failure']=dict(path=str(history),sha256=sha(history),
                    facts='历史本机rint_cld对float16全域65536项出现2050项数值不符，例0.5应0而输出1；低精度nearbyint直接调用不支持。',
                    scope='历史版本负证据，不把旧debug延迟当本轮性能。当前版本实现有改变时可作一次小范围语义核验；不默认intrinsic等于一条硬件指令。')
            evidence['diagnosis']+=('起点舍入采用FP32大常数法和符号位恢复。优先检查4096小块产生大量program的搬运组织、片上缓冲存活和宿主层次。'
                '可尝试本批已核验的40核流式处理及完整块/尾块分离；保持half-to-even、负零、NaN/Inf、decimals和真实stride语义。'
                '增大block须检查实际UB分配，不能将FP16算术技巧直接套到BF16。')
        if op in REMAINDER:
            evidence['unresolved'] = old['status']
            extra = old.get('supplementary') or {}
            if extra.get('path'): evidence['supplementary_failures'] = read(extra['path'])
            evidence['diagnosis'] += ('这项参考或数值仍存在缺口，先严格修复原语义，不能通过修改容差或参考规避。'
                                     'uniform禁止重写原生CANN参考以掩盖下界舍入问题；fractional没有原生接口不能伪造。')
            inventory=BATCH/'reference-probes/cann-9-interface-inventory-v1.json'
            if inventory.exists():
                value=read(inventory)
                evidence['additional_CANN_interface_inventory']=dict(path=str(inventory),sha256=sha(inventory),
                    exports=value['exports'],graph_interfaces=[{k:r[k] for k in ['name','path','sha256']}
                        for r in value['graph_ops']],
                    scope='当前安装版本的只读声明/符号证据，不证明设备支持，也不计作合格参考')
            if op=='expm1_':
                alternate=BATCH/'reference-probes/native-foreach-expm1-v1/numerics.json'
                if alternate.exists():
                    value=read(alternate)
                    evidence['native_foreach_expm1_probe']=dict(path=str(alternate),sha256=sha(alternate),
                        status=value['status'],passed=value['passed'],total=value['total'],
                        qualified_CANN=value['qualified_CANN'],performance_measured=value['performance_measured'],
                        cases=[{k:r.get(k) for k in ['dtype','alias_output','strided','passed','error']}
                               for r in value['cases']])
                    evidence['diagnosis']+=('ForeachExpm1适配只属于参考端资格探测。即使小样本通过，也需完整合同、补测和路径核验后才可作新参考；'
                        '该CANN/C++实现不能进入严格Triton候选，不把原生数值问题当作候选必须复制的缺陷。')
            if op=='cumulative_trapezoid':
                conflict=PROJECT/'operators/campaigns/reference-closure-20261031/analysis/cumulative_trapezoid/agent-context.json'
                if conflict.exists():
                    prior=read(conflict)
                    evidence['historical_reference_conflict']=dict(path=str(conflict),sha256=sha(conflict),
                        facts=prior['execution_guidance']['reference_conflict'],
                        policy='仅引用参考冲突事实；历史CANN候选建议无效。不得改变全局确定性模式或容差。')
                evidence['diagnosis']+=('已有补测失败集中在float16[1024,128]沿dim0的两组fresh输入。'
                    '当前源码对所有float16逐步half舍入；不要接受docstring关于参考扫描顺序的自述。'
                    '应分别检查末轴/非末轴扫描的累加精度、相邻加法和缩放的舍入位置；'
                    '可测试非末轴FP32累加这一假设，保留原输入和容差。')
            if op=='linalg_inv_ex':
                conflict=PROJECT/'operators/campaigns/remaining-nine-20261030/analysis/linalg_inv_ex/reference-mathematical-conflict.json'
                if conflict.exists():
                    evidence['historical_reference_conflict']=dict(path=str(conflict),sha256=sha(conflict),facts=read(conflict),
                        policy='默认合同保留；不得按输入值或用例硬编码逆矩阵/info，也不得以CPU或CANN计算替代Triton。')
                evidence['diagnosis']+=('旧源码关于CPU参考转置及消元次序的docstring并非证明。'
                    '检查实际矩阵布局、除法/倒数乘法、FMA和消元结合顺序；原参考对重复行矩阵返回info=0和很大有限值。'
                    '保留奇异输入失败证据，不能放宽容差、改变奇异语义或硬编码结果来凑通过。')
                arithmetic=PROJECT/'operators/campaigns/reference-closure-20261031/analysis/linalg_inv_ex/offline-layout-v2.json'
                if arithmetic.exists():
                    history=read(arithmetic)['rows']
                    groups=sorted({(r['case'],r['matrix']) for r in history})
                    evidence['offline_arithmetic_history']=dict(path=str(arithmetic),sha256=sha(arithmetic),
                        scope='已有离线FP32运算顺序假设；不是本轮NPU结果，不把个别固定输入常量写进候选',
                        cases=[dict(case=name,matrix=index,
                            attempted=sum(r['case']==name and r['matrix']==index for r in history),
                            matched=sum(r['case']==name and r['matrix']==index and r.get('matched',False) for r in history))
                            for name,index in groups])
                    evidence['diagnosis']+=('上述历史已覆盖直接/转置、逐项除法/倒数乘法、LU和求解的FMA组合等。'
                        '不要重复无新机制的相同排列。若研究LAPACK一致性，可检查实际分块/递归分解及三角求解顺序，'
                        '形成通用Triton算法；不能按奇异用例预存参考值。')
        return evidence
    def prepare_paired(self,op,seed):
        if self.q.records[op]['reference']!='qualified_CANN':return
        folder=BATCH/'analysis'/op/('paired-'+sha(seed)[:16])
        if (folder/'summary.json').exists():
            assert read(folder/'summary.json')['spec']['source_sha256']==sha(seed)
            return
        if folder.exists():raise self.module.Attention('独立采集存在未完成原件，不覆盖重启：'+str(folder))
        previous=self.previous[op]['cann'];cases=read(previous['path']).get('per_workload',[])
        timing=[r for r in cases if r.get('phase')=='timing' and r.get('latency_ms') and r.get('reference_latency_ms')]
        if not timing:return
        worst=min(timing,key=lambda r:r['reference_latency_ms']/r['latency_ms'])['uuid']
        leaf=BATCH/'catalog/ops'/op
        case=next(v for v in (json.loads(x) for x in (leaf/'timing.jsonl').read_text().splitlines() if x.strip()) if v['name']==worst)
        spec=folder.with_name(folder.name+'-spec.json')
        payload=dict(operator=op,source=str(seed),source_sha256=sha(seed),leaf=str(leaf),case=case,
                     reference_kind='frozen_qualified_CANN',starting_result=previous['path'])
        if spec.exists():assert read(spec)==payload
        else:save(spec,payload)
        self.cmd('paired-'+op+'-'+sha(seed)[:8],[PY,HERE/'paired.py','--spec',spec,'--output',folder],1800)
        assert (folder/'summary.json').exists()
    def refresh_context(self,op,attempt,seed,overlay):
        """仅在本搜索工作区尚未启动时，固化当前直接采集的证据与角色。"""
        workspace=self.q.runtime/'kg-runs'/(op+'-'+BATCH.name+'-attempt-'+str(attempt))
        if workspace.exists():raise self.module.Attention('已启动工作区不改角色：'+str(workspace))
        evidence=self.context(op,seed)
        save(BATCH/'analysis'/op/('search-'+str(attempt)+'.json'),evidence)
        content='\n\n# HG-KDA严格Triton夜间任务\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n'
        roles={}
        for role in (overlay/'kernelgen/.kernelgen/agents').glob('*.md'):
            if role.stem in {'kernel-coder','kernel-knowledge-coder','kernel-optimize','kernel-profile-analyzer','kernel-knowledge-profile-analyzer'}:
                base=role.read_text().split('\n\n# HG-KDA严格Triton夜间任务')[0]
                role.write_text(base+content)
            roles[role.name]=sha(role)
        record=read(overlay/'strict-night-overlay.json');record['roles']=roles
        record['adapter_sha256']=sha(overlay/'kernelgen/tools/kernelgen_server_adapter.py')
        record['admission_sha256']=sha(overlay/'kernelgen/tools/hgkda_strict_admission.py')
        save(overlay/'strict-night-overlay.json',record)
        save(BATCH/'analysis'/op/('overlay-'+str(attempt)+'.json'),record)
    def overlay(self,op,attempt,seed):
        output = ROOT/'runtime'/BATCH.name/(op+'-search-'+str(attempt))/'sources'
        if output.exists():
            assert (output/'strict-night-overlay.json').exists(), '覆盖层准备未完成'
            return output
        self.cmd('overlay-'+op+'-'+str(attempt),[PY,TOOLS/'build-kg-overlay.py','--output',output,'--native-context'],600)
        shutil.copyfile(HERE/'admission.py',output/'kernelgen/tools/hgkda_strict_admission.py')
        adapter = output/'kernelgen/tools/kernelgen_server_adapter.py'
        text = adapter.read_text()
        for name in ['preflight_bundle','evaluate_bundle']:
            start=text.index('def '+name+'(')
            i=text.index('    from kernelgen_client.http import ',start)
            text=text[:i]+('    from kernelgen.tools.hgkda_strict_admission import admit_bundle\n'
                            '    admit_bundle(bundle, context)\n\n')+text[i:]
        adapter.write_text(text)
        with adapter.open('a') as stream:
            stream.write('''

# 本批同内容评测只计一次；历史原件保留。
_strict_night_evaluate = evaluate_bundle
def evaluate_bundle(bundle, context, *, run_control=None):
    from kernelgen.tools.hgkda_strict_admission import cached_evaluate
    return cached_evaluate(_strict_night_evaluate, bundle, context,
                           _settings_from_context(context), run_control)
''')
        # 复用已有有界诊断；只改本批overlay。
        shutil.copyfile(TOOLS/'next_phase/debug_budget.py', output/'kernelgen/tools/hgkda_debug_budget.py')
        server = output/'kernelgen/mcp_server/server.py'; text=server.read_text()
        needle='    workspace = workspace_from_env()\n    try:\n        return submit_workspace_debug_job('
        replacement=('    workspace = workspace_from_env()\n'
                     '    from kernelgen.tools.hgkda_debug_budget import admit\n'
                     '    admission = admit(workspace, request.timeout_seconds)\n'
                     "    if admission['status'] != 'ADMITTED':\n        return admission\n"
                     "    request = request.model_copy(update={'timeout_seconds': admission['timeout_seconds']})\n"
                     '    try:\n        return submit_workspace_debug_job(')
        assert text.count(needle)==1;server.write_text(text.replace(needle,replacement))
        evidence = self.context(op,seed)
        save(BATCH/'analysis'/op/('search-'+str(attempt)+'.json'),evidence)
        content='\n\n# HG-KDA严格Triton夜间任务\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n'
        roles={}
        for role in (output/'kernelgen/.kernelgen/agents').glob('*.md'):
            if role.stem in {'kernel-coder','kernel-knowledge-coder','kernel-optimize','kernel-profile-analyzer','kernel-knowledge-profile-analyzer'}:
                # 用本批精简上下文代替全部历史混合后端模板，保留原角色和严格规则。
                base=role.read_text().split('\n\n# HG-KDA 角色扩展')[0]
                role.write_text(base+content)
            roles[role.name]=sha(role)
        save(output/'strict-night-overlay.json',dict(operator=op,attempt=attempt,roles=roles,
                 admission_sha256=sha(output/'kernelgen/tools/hgkda_strict_admission.py'),
                 adapter_sha256=sha(adapter),seed_sha256=sha(seed)))
        save(BATCH/'analysis'/op/('overlay-'+str(attempt)+'.json'),read(output/'strict-night-overlay.json'))
        return output
    def search(self,op,attempt):
        while len(self.q.plan['passes'])<attempt:
            self.q.plan['passes'].append(dict(self.q.plan['passes'][0]))
        row=self.state['operators'].setdefault(op,dict(attempts=[]))
        key=op+':search-'+str(attempt)
        if self.state['jobs'].get(key,{}).get('status')=='completed':return
        workspace=self.q.runtime/'kg-runs'/(op+'-'+BATCH.name+'-attempt-'+str(attempt))
        if not workspace.exists():
            self.process_proposals()
            ready=(row.get('best') or {})
            if (ready.get('qualification')=='passed' and ready.get('reference')=='qualified_CANN'
                    and ready.get('search_score',0)>.8 and ready.get('compiled_status')=='verified'
                    and ready.get('review_status')=='reviewed_strict_triton'):
                self.state['jobs'][key]=dict(status='completed',result=dict(
                    status='verified_target_retained_no_remeasure',source=ready['source'],
                    source_sha256=ready['source_sha256']),finished_at=self.module.now())
                self.persist()
                return
        seed=Path((row.get('best') or {}).get('source', BATCH/self.q.records[op]['seed']))
        if (row.get('best') or {}).get('host_glue'):
            # 宿主桥只在人工绑定的独立提案中使用；CC搜索保持自包含Python+Triton。
            seed=BATCH/self.q.records[op]['seed']
        self.state.update(status='running',current_job=key)
        self.state['jobs'][key]=dict(status='running',seed=str(seed),source_sha256=sha(seed));self.persist()
        self.prepare_paired(op,seed)
        overlay=self.overlay(op,attempt,seed)
        # native_search的exports不覆盖这三个变量，docker exec显式传入。
        # 变量由独立run-kg脚本导出，宿主环境不泄露任何模型凭据。
        wrapper=ROOT/'runtime'/BATCH.name/'run-kg-strict.sh'
        if not wrapper.exists():
            script=(PROJECT/'tools/evaluation/run-kg-operator-910b.sh').read_text()
            # 原脚本通过自身位置定位repo；复制版必须显式使用固定repo。
            script=script.replace('repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)',
                                  'repo='+shlex.quote(str(self.q.repo)))
            script=script.replace('"$(dirname "$0")/wait-kg-worker.py"',
                                  shlex.quote(str(PROJECT/'tools/evaluation/wait-kg-worker.py')))
            script=script.replace('set -euo pipefail','set -euo pipefail\nexport HGKDA_STRICT_BATCH='+shlex.quote(str(BATCH))+
                 '\nexport HGKDA_RUNTIME_AUDIT='+shlex.quote(str(TOOLS/'strict_triton/runtime_audit.py')))
            wrapper.parent.mkdir(parents=True,exist_ok=True);wrapper.write_text(script)
        # 只替换native helper源码中的启动脚本路径，保留它的请求身份/超时保护。
        helper=ROOT/'runtime'/BATCH.name/'native_search.py'
        if not helper.exists():
            text=(TOOLS/'next_phase/native_search.py').read_text().replace(
                "self.q.project/'tools/evaluation/run-kg-operator-910b.sh'",repr(str(wrapper)))
            helper.write_text(text)
        method=module('strict_night_native_'+str(attempt),helper).native
        self.wait()
        try:
            workspace=method(self,op,'attempt-'+str(attempt),seed,overlay,2400)
        except self.module.Attention:
            if not (BATCH/('budget-'+op+'-attempt-'+str(attempt)+'.json')).exists():raise
            workspace=self.q.runtime/'kg-runs'/(op+'-'+BATCH.name+'-attempt-'+str(attempt))
            stop=module('night_stop',TOOLS/'next_phase/stop_worker.py')
            stop.stop(workspace,BATCH/'analysis'/op/('own-worker-stop-'+str(attempt)+'.json'))
            save(BATCH/'analysis'/op/('cleanup-'+str(attempt)+'.json'),stop.cleanup_debug(workspace))
            self.wait()
        result=self.collect(op,attempt,workspace)
        row['attempts'].append(result)
        self.refresh_existing_scores()
        self.verify_reviewed_best()
        self.state['jobs'][key].update(status='completed',result=result,finished_at=self.module.now());self.persist()
    def collect(self,op,attempt,workspace):
        from measurements import ledger,computation_identity,baseline_score
        candidates=[];record=dict(workspace=str(workspace),attempt=attempt,candidates=[])
        archive=BATCH/'search-archives'/op/('search-'+str(attempt)+'.tar.gz')
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True);identities={}
            with tarfile.open(archive,mode='w:gz') as stream:
                for path in sorted(workspace.rglob('*')):
                    if not path.is_file() or path.is_symlink():continue
                    relative=path.relative_to(workspace)
                    if any(v in relative.parts for v in ['__pycache__','.git','compiler-cache','triton-cache']):continue
                    # 保留实际角色、源代码、请求、结果、分析与原始运行日志；不收集运行时设置文件。
                    if path.suffix not in {'.py','.md','.json','.jsonl','.log','.txt'}:continue
                    if 'settings' in path.name or path.name in {'.env','credentials.json'}:continue
                    identities[str(relative)]=dict(sha256=sha(path),bytes=path.stat().st_size)
                    stream.add(path,arcname=str(relative),recursive=False)
            save(archive.with_name(archive.name+'.manifest.json'),dict(workspace=str(workspace),files=identities,
                 archive_sha256=sha(archive),scope='已结束工作区的实际角色/模型日志/请求/结果，运行时设置和缓存除外'))
        record['archive']=dict(path=str(archive),sha256=sha(archive))
        old_sha=self.previous[op]['source_sha256']
        original_id=computation_identity(BATCH/self.q.records[op]['seed'])
        seen={r['source_sha256'] for a in self.state['operators'][op]['attempts'] for r in a.get('candidates',[])}
        seen_computations={original_id}|{computation_identity(Path(r['source']))
            for a in self.state['operators'][op]['attempts'] for r in a.get('candidates',[])}
        # 独立提案已经实测；后续KG即使再次提交相同AST，也不能把较高复测分数算成新改进。
        seen_computations.update(computation_identity(Path(r['source']))
            for r in self.state['operators'][op].get('proposals',[]))
        for result in sorted(workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/result.json')):
            source=result.with_name('main.py'); v=read(result)
            if not source.exists() or v.get('status')!='PASSED' or v.get('is_hack') or sha(source)==old_sha:continue
            computation=computation_identity(source)
            if computation in seen_computations:continue
            if v.get('num_passed')!=v.get('num_workloads') or not v.get('num_workloads'):continue
            # 同源码只保留第一次完整评测，不按重复测量中的高分择优。
            digest=sha(source)
            if digest in seen:continue
            seen.add(digest)
            seen_computations.add(computation)
            solution=read(result.with_name('solution.json')).get('sources',[])
            if len(solution)!=1 or solution[0].get('content')!=source.read_text(encoding='utf8'):
                raise self.module.Attention('KG记录与实际源码不一致')
            measurements=ledger(workspace,source)
            if measurements['status']!='passed':continue
            candidates.append((measurements['score'],source,result))
        # 不重测KG已经完整评测过的同一源码；按CANN结果排序后补验原语义。
        for score,source,result in sorted(candidates,reverse=True):
            out=BATCH/'candidates'/op/(sha(source)[:16]);out.mkdir(parents=True,exist_ok=True)
            target=out/'main.py'
            if target.exists():assert sha(target)==sha(source)
            else:shutil.copyfile(source,target)
            shutil.copyfile(result,out/'search-result.json')
            entry=dict(source=str(target),source_sha256=sha(target),search_result=str(result),search_score=score,
                       search_round_score=read(result)['geo_mean'],
                       provenance='KG/HG-KDA与CC/DeepSeek生成；Codex诊断、编排与门禁',
                       reference=self.q.records[op]['reference'])
            # 至多验证本次前三个不同源码；未选项保留KG完整结果。
            if len(record['candidates'])>=3:break
            existing_pt=result if self.q.records[op]['reference']=='original_PyTorch_only' else None
            rejection=self.manual_rejection(op,target)
            if rejection:
                entry.update(qualification='manual_review_rejected',manual_review_rejection=rejection,compiled_status='not_qualified')
            else:entry.update(self.validate(op,target,out,existing_pytorch_result=existing_pt))
            record['candidates'].append(entry)
            row=self.state['operators'][op]
            baseline=baseline_score(self.previous[op])
            best_score=(row.get('best') or {}).get('search_score',baseline)
            if entry['qualification']=='passed' and score>best_score:
                row['best']=entry
            # 其余源码已有KG完整CANN结果，保留原件；只给选中版本追加原语义/补测，避免无必要的重复工作。
            if entry['qualification']=='passed' and score>baseline:break
        record['status']='candidates_evaluated' if record['candidates'] else 'no_new_valid_candidate'
        return record
    def validate(self,op,source,out,existing_pytorch_result=None):
        original=BATCH/'original-catalog/ops'/op
        from validation_status import supplement_status
        extras={}
        for label,leaf in [('pytorch',original),('cann',BATCH/'catalog/ops'/op)]:
            if label=='cann' and self.q.records[op]['reference']!='qualified_CANN':continue
            output=out/('extra-'+label+'.json')
            if not output.exists():
                try:self.cmd('extra-'+op+'-'+sha(source)[:8]+'-'+label,
                    [PY,TOOLS/'cann_priority/validate_cann.py','--operator',op,'--source',source,'--leaf',leaf,'--output',output],1200)
                except RuntimeError:
                    if not output.exists():raise
            extras[label]=dict(path=str(output),status=supplement_status(op,read(output),sha(source)))
        if any(value['status']!='passed' for value in extras.values()):
            return dict(qualification='failed',extras=extras,pytorch_result=None,pytorch_score=None,
                        review_status='pending_Codex_manual_review',compiled_status='not_qualified')
        formal=Path(existing_pytorch_result) if existing_pytorch_result else out/'pytorch/strict-PyTorch-1.result.json'
        if existing_pytorch_result:
            assert self.q.records[op]['reference']=='original_PyTorch_only'
            for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
                assert (BATCH/'catalog/ops'/op/name).read_bytes()==(original/name).read_bytes()
            request=formal.with_name(formal.name.replace('.result.json','.request.json'))
            if formal.name.endswith('.result.json') and request.exists():
                sources=read(request)['implementation']['sources']
            else:
                sources=read(formal.with_name('solution.json'))['sources']
            assert len(sources)==1 and sources[0]['content']==Path(source).read_text(encoding='utf8')
        elif not formal.exists():
            self.cmd('pytorch-'+op+'-'+sha(source)[:8],[PY,TOOLS/'cann_priority/evaluate.py','--operator',op,
                '--source',source,'--leaf',original,'--output',out/'pytorch','--server',self.q.server,
                '--label','strict-PyTorch','--kind','strict_Triton_candidate'])
        value=read(formal)
        qualification='passed' if (value.get('status')=='PASSED' and not value.get('is_hack') and
                         value.get('num_passed')==value.get('num_workloads') and
                         all(v['status']=='passed' for v in extras.values())) else 'failed'
        return dict(qualification=qualification,extras=extras,pytorch_result=str(formal),
                    pytorch_score=value.get('geo_mean'),review_status='pending_Codex_manual_review',
                    compiled_status='pending_selected_verification')
    def report(self):
        accepted=sum(r['accepted'] for r in self.previous.values());provisional=[];verified=[]
        for op,row in self.state.get('operators',{}).items():
            best=row.get('best') or {}
            if best.get('reference')=='qualified_CANN' and best.get('search_score',0)>.8 and not self.previous[op]['accepted']:
                provisional.append(op)
                if (best.get('qualification')=='passed' and best.get('compiled_status')=='verified' and
                        best.get('review_status')=='reviewed_strict_triton'):verified.append(op)
        save(BATCH/'progress.json',dict(status=self.state['status'],current_job=self.state.get('current_job'),
              frozen_accepted=accepted,new_numerically_qualified=provisional,new_backend_verified=verified,
              accepted=accepted+len(verified),scope='新增只有完成原语义/补测、人工审查与实际TTIR/npubin/设备路径核验后才计数；最终全量证据核验另做'))
        lines=['# 严格Triton夜间优化进展','',f'冻结起点：{accepted}/30；本轮已核验新增{len(verified)}项，累计{accepted+len(verified)}/30；状态：{self.state["status"]}。',
               '', '| 算子 | 起点相对CANN | 新候选相对CANN | 原语义及补测 | 后端审查 |',
               '| --- | ---: | ---: | --- | --- |']
        for op in PRIORITY+REMAINDER:
            old=self.previous[op];best=(self.state.get('operators',{}).get(op) or {}).get('best') or {}
            a=old.get('cann',{}).get('score');b=best.get('search_score') if best.get('reference')=='qualified_CANN' else None
            lines.append('| '+op+' | '+(f'{a:.4f}×' if a is not None else '参考待解决')+' | '+
                         (f'{b:.4f}×' if b is not None else '待完成')+' | '+best.get('qualification','待完成')+' | '+
                         best.get('review_status','待完成')+' |')
        (BATCH/'夜间优化进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        # 离线汇总只读取已结束工作区与已核验源码，不触发设备评测。
        module('strict_night_delivery_snapshot',HERE/'report.py').main()
    def execute(self):
        self.state.pop('error',None)
        self.state.update(status='running');self.persist()
        for attempt in [1,2]:
            for op in PRIORITY+REMAINDER:
                best=(self.state['operators'].get(op) or {}).get('best') or {}
                if attempt>1 and (best.get('search_score',0)>.85 or op in {'uniform','fractional_max_pool3d'}):continue
                self.search(op,attempt)
        self.wait();self.state.update(status='search_completed_pending_review',current_job=None,finished_at=self.module.now());self.persist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare()
    if a.prepare:print(BATCH);return
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=Runner()
        try:runner.execute()
        except Exception as error:
            runner.state.update(status='requires_review',error=type(error).__name__+': '+str(error));runner.persist();raise


if __name__=='__main__':main()
