"""仅修复已明确的参考包装ABI错误；保留原始失败及冻结合同。"""
import hashlib
import json
from pathlib import Path


def process(r):
    from run import PY,quote,save_json
    op='hardswish_out'
    old=r.state.get('references',{}).get(op) or {}
    if old.get('status')!='RUNTIME_ERROR':return
    failed=json.loads(Path(old['result']).read_text(encoding='utf8'))
    if 'expected positional or keyword out, got keyword-only out' not in failed.get('log',''):
        return
    out=r.batch/'references'/op/'abi-v2'
    source=out/'main.py'
    old_source=Path(old['source']);content=old_source.read_text(encoding='utf8')
    assert hashlib.sha256(old_source.read_bytes()).hexdigest()==old['source_sha256']
    assert content.count('def run(x, *, out):')==1
    content=content.replace('def run(x, *, out):','def run(x, out):')
    out.mkdir(parents=True,exist_ok=True)
    if source.exists() and source.read_text(encoding='utf8')!=content:raise r.module.Attention('ABI修复源码已改变')
    if not source.exists():source.write_text(content,encoding='utf8')
    provenance_path=Path(old.get('provenance') or old_source.parent/'provenance.json')
    provenance=json.loads(provenance_path.read_text(encoding='utf8'))
    provenance.update(binding_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      prior_failed_source_sha256=old['source_sha256'],correction='out改为原合同的可位置/关键字参数；库调用和测试合同不变')
    save_json(out/'provenance.json',provenance)
    here=Path(__file__).resolve().parent
    def evaluate():
        r.q.run('reference-evaluate-hardswish-abi-v2',quote([PY,here/'evaluate_reference.py',
            '--operator',op,'--source',source,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
            '--output',out/'formal','--server',r.q.server,'--label','CANN-async-python-abi-v2']),2100)
        path=out/'formal/CANN-async-python-abi-v2-1.result.json'
        value=json.loads(path.read_text(encoding='utf8'))
        return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),
                    num_workloads=value.get('num_workloads'),result=str(path),source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    reference_kind='CANN_native_aclnn',provenance=str(out/'provenance.json'))
    result=r.job('reference-evaluate:hardswish_out:abi-v2',evaluate)
    if result and result.get('status')=='PASSED':
        gates=out/'extra-checks.json'
        qualification=r.job('reference-extra:hardswish_out:abi-v2',lambda:r.supplementary(op,source,gates,'reference-extra-hardswish-abi-v2'))
        result.update(extra_checks=str(gates),supplementary_status=qualification)
    if result:
        r.state.setdefault('reference_history',{}).setdefault(op,[]).append(old)
        r.state['references'][op]=result;r.save()
