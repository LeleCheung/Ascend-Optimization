"""在扫描KG工作区创建前，冻结参考不稳定反例并提供限额事实摘要。"""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=PROJECT/'operators/campaigns/next25-20261029-v2/analysis/cumulative_trapezoid/reference-stability-replay-v1'
    output=BATCH/'analysis/cumulative_trapezoid/reference-stability-replay-v1';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ['input-fp16-256x8.bin','default.json','deterministic.json','provenance.json','reproducer.py']:
        old=source/name;new=output/name
        if new.exists():assert new.read_bytes()==old.read_bytes()
        else:new.write_bytes(old.read_bytes())
        records.append(dict(path=str(new),sha256=sha(new),origin=str(old)))
    meta=read(output/'provenance.json');default=read(output/'default.json');det=read(output/'deterministic.json')
    assert meta['input_sha256']==default['input_sha256']==det['input_sha256']==sha(output/'input-fp16-256x8.bin')
    assert sha(output/'reproducer.py')==meta['script_sha256']
    assert default['deterministic'] is False and det['deterministic'] is True
    assert default['empty_tolerance_intersections']>0 and det['empty_tolerance_intersections']==0
    values=default['values'];lower=max(x-(.015+.015*abs(x)) for x in values);upper=min(x+(.015+.015*abs(x)) for x in values)
    assert lower>upper
    leaf=BATCH/'catalog/ops/cumulative_trapezoid';seed=BATCH/'starting-points/cumulative_trapezoid.py'
    original=read(BATCH/'bootstrap.json')['cumulative_trapezoid'];assert sha(seed)==original['sha256']
    context=dict(schema='hgkda-reference-closure-scan-context-v1',status='historical_reference_conflict_verified',
        spec=dict(operator='cumulative_trapezoid',source_sha256=sha(seed),scope='相同固定环境的已冻结反例；非本轮候选profiling'),
        source=dict(path=str(seed),sha256=sha(seed)),
        execution_guidance=dict(goal='优化真实输入和通用布局，争取原KGS合同完整通过；候选不能改变全局确定性模式',
            reference_conflict=dict(shape=default['shape'],dtype=default['dtype'],dim=default['dim'],dx=default['dx'],
                input_sha256=default['input_sha256'],position=default['position'],repeated_reference_values=values,
                common_lower=lower,common_upper=upper,empty_intersections=default['empty_tolerance_intersections'],
                deterministic_consistent=True,scope='这个补测输入的默认参考不存在满足全部重复结果的固定输出；不表示每个原KGS case都必然失败'),
            instructions=['先提交可运行完整候选，保留原合同、FP16舍入步骤、stride和当前stream语义',
                '若完整合同通过仍须补测冻结反例，不能选择恰好通过的一次结果',
                '检查加法、dx缩放、累加和dtype转换顺序；可以优化连续末轴/非末轴及布局处理',
                'CANN扫描可作真实厂商原语；不调用PyTorch目标算子绕过候选',
                '确认默认参考矛盾无法由候选解决时输出可复现原因，确定性条件另立报告，不伪造全部达标']),
        limits=['历史0.812×在官方确定性条件下；不是默认条件稳定交付','默认同输入重复漂移是参考事实，不能直接归因为候选错误'],
        provenance=dict(evidence=records,contract_sha256={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']}))
    target=BATCH/'analysis/cumulative_trapezoid/agent-context.json'
    if target.exists():assert read(target)==context,'已有扫描上下文不覆盖'
    else:target.write_text(json.dumps(context,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    print(json.dumps(dict(status='prepared',context=str(target),sha256=sha(target),common_interval_empty=True)))
if __name__=='__main__':main()
