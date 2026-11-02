"""从已冻结的四项组件诊断生成可读比较，不执行设备请求。"""
import hashlib
import json
from pathlib import Path


def main():
    batch=Path(__file__).resolve().parents[2]
    source=batch/'component-diagnostics/expm1-components-v1/result.json'
    value=json.loads(source.read_text(encoding='utf8'))
    assert value['status']=='measured' and value['qualified_CANN'] is False
    assert value['formal_performance_measured'] is False
    rows={row['name']:row for row in value['rows']}
    comparison=[]
    for size in ['small','large']:
        candidate=rows[size+'-candidate']
        reference=rows[size+'-original_PyTorch']
        assert candidate['case']==reference['case']
        comparison.append(dict(size=size,case=candidate['case'],
            candidate_device_us=candidate['device_task_union_us'],
            pytorch_device_us=reference['device_task_union_us'],
            candidate_to_pytorch_device_ratio=candidate['device_task_union_us']/reference['device_task_union_us'],
            candidate_host_median_us=candidate['host_call_us']['median'],
            pytorch_host_median_us=reference['host_call_us']['median']))
    record=dict(status='derived_from_completed_diagnostics',
        input=str(source.relative_to(batch)),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=comparison,
        finding='小输入设备耗时接近，宿主路径有差距；大输入候选设备任务为原PyTorch约4.38倍，需重点优化设备搬运/计算/循环组织。',
        limitations=['四项是独立诊断，带profiler设备时间不与宿主样本相加减。',
            '没有Vector/MTE流水指标，不能仅凭总设备时间认定内存受限或计算受限。',
            '不是新的正式性能测量；原PyTorch底层CANN路径不代表额外精度检查已通过。'])
    output=Path(__file__).with_name('components-v1-summary.json')
    raw=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    if output.exists():assert output.read_text(encoding='utf8')==raw
    else:output.write_text(raw,encoding='utf8')
    print(json.dumps(record,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
