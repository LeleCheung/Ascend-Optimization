"""保留完整块与启动桥的已测结果，区分正确、改进和实际后端核验。"""
import hashlib
import json
from pathlib import Path


def main():
    batch=Path(__file__).resolve().parents[2]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    parent=batch/'candidates/expm1_/7696add46b061fd8'
    original=parent/'formal-reference/result.json'
    if not original.exists():
        original=parent/'search-result.json'
    baseline=read(original)['geo_mean']
    evidence={original.relative_to(batch).as_posix():sha(original)}
    rows=[]
    for name in ['codex-fulltiles-expm1-v2','codex-vectorcall-expm1-v3']:
        folder=batch/'proposals/expm1_'/name
        source=folder/'main.py'
        spec=read(folder/'spec.json')
        assert sha(source)==spec['source_sha256']
        result=folder/'pytorch/Codex-strict-PyTorch-1.result.json'
        value=read(result)
        assert value['status']=='PASSED' and value['num_passed']==value['num_workloads']==30
        extra=folder/'extra-pytorch.json'
        supplement=read(extra)
        assert supplement['source_sha256']==sha(source) and supplement['status']=='passed'
        assert supplement['passed']==supplement['total']==63
        rows.append(dict(name=name,source_sha256=sha(source),PyTorch=value['geo_mean'],
            formal='30/30',supplement='63/63',exceeds_previous_best=value['geo_mean']>baseline,
            selected_for_delivery=False,backend_verification='未选中，未追加代表输入后端采集',
            host_glue=bool(spec.get('host_glue'))))
        for file in [source,folder/'spec.json',folder/'review.json',folder/'runtime-audit.json',result,extra]:
            evidence[file.relative_to(batch).as_posix()]=sha(file)
    context=batch/'analysis/expm1_/search-2.json'
    actual=read(context)
    assert len(actual['execution_components']['rows'])==4
    evidence[context.relative_to(batch).as_posix()]=sha(context)
    record=dict(status='measured_valid_not_selected',baseline_pytorch_speedup=baseline,
        rows=rows,evidence_sha256=evidence,
        host_vs_device_parent_ratio=rows[1]['PyTorch']/rows[0]['PyTorch'],
        interpretation='两个新稿数值通过；完整块与启动桥均未超过原最佳0.561×，保留原版本。第二轮已携带四项执行诊断，从原最佳源码继续。',
        scope='各源码只使用各自已有的一次完整正式测量；近邻分数不作统计显著性或单因素因果结论；不计CANN资格。')
    output=Path(__file__).with_name('fulltiles-vectorcall-v1-summary.json')
    raw=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    if output.exists():assert output.read_text(encoding='utf8')==raw
    else:output.write_text(raw,encoding='utf8')
    print(json.dumps(dict(status=record['status'],baseline=baseline,rows=rows),ensure_ascii=True))


if __name__=='__main__':
    main()
