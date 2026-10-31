"""把本批已完成的完整结果加入模型上下文；成对采集原件及源码作用域保持明确。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def write(op,context):
    state=read(BATCH/'next-state.json');row=state['operators'][op]
    evidence=[]
    for label,record in [('historical_start',row.get('starting_point')),('best_current',row.get('best'))]:
        if not record:continue
        raw=read(Path(record['result']))
        evidence.append(dict(label=label,source=record['source'],source_sha256=record['source_sha256'],
            backend=record.get('backend','Triton'),result=record['result'],status=raw['status'],speedup=raw.get('geo_mean'),
            timing_cases=[{k:x.get(k) for k in ['uuid','latency_ms','reference_latency_ms','speedup']} for x in raw.get('per_workload',[]) if x.get('phase')=='timing']))
    failures=[]
    for path in (BATCH/'proposals'/op).glob('*/assessment/formal/*.result.json'):
        raw=read(path)
        if raw.get('status')=='PASSED':continue
        failures.append(dict(result=str(path),status=raw.get('status'),passed=raw.get('num_passed'),total=raw.get('num_workloads'),
            cases=[{k:(str(x.get(k) or '')[-1600:] if k=='log' else x.get(k)) for k in ['uuid','status','log']}
                   for x in raw.get('per_workload',[]) if x.get('status')!='PASSED'][:3]))
    context['execution_guidance']=dict(current_full_call_evidence=evidence,current_failed_attempts=failures,
        constraints=['完整调用与独立profiling时间分开；关注大输入的实际落后，不能只优化小输入宿主路径',
                     'expm1必须保持接近零时相对精度与CPU数学参考零符号；不能靠反复原地调用后的数值退化来跳过目标计算',
                     'linalg保留重复行输入、原容差和info；不用PyTorch或CPU目标函数伪装NPU实现；编译失败原件用于避免重复无效路径'],
        attribution='本批Codex整合的已完成KGS结果；机制假设由Agent用同输入源码/IR/指标继续验证')
    folder=BATCH/'analysis'/op;folder.mkdir(parents=True,exist_ok=True)
    target=folder/'agent-context-current-evidence.json'
    if target.exists():raise FileExistsError(target)
    raw=(json.dumps(context,ensure_ascii=False,indent=2)+'\n').encode('utf8');target.write_bytes(raw)
    pointer=folder/'latest-context.json';temp=pointer.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n',encoding='utf8');temp.replace(pointer)
    print('WROTE '+str(target),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch-expm1',action='store_true');a=p.parse_args()
    if not (BATCH/'analysis/linalg_inv_ex/agent-context-current-evidence.json').exists():
        write('linalg_inv_ex',dict(schema='hgkda-current-attempt-context-v1',status='formal_results_only',
            limits=['此文件只汇总正式结果，尚未提供新的成对metrics或IR；原生控制器会另外加入当前源码绑定的inverse probe']))
    if a.watch_expm1:
        original=BATCH/'analysis/expm1_/agent-context.json'
        deadline=time.monotonic()+5400
        while time.monotonic()<deadline:
            if original.exists():
                write('expm1_',read(original));return
            time.sleep(1)
        raise RuntimeError('未等待到本批真实成对上下文；不生成伪采集')


if __name__=='__main__':main()
