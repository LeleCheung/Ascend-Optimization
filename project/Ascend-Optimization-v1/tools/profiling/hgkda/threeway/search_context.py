"""把本轮完整结果及被否定的改法交给后续KG搜索，不伪装成新metrics。"""
import hashlib
import json
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[4]
    batch=project/'operators/campaigns/threeway-20261030'
    state=json.loads((batch/'next-state.json').read_text(encoding='utf8'))
    for op in ['cross_out','nan_to_num_']:
        best=state['operators'][op]['best'];source=Path(best['source'])
        case=json.loads((batch/'catalog/ops'/op/'timing.jsonl').read_text().splitlines()[0])
        rows=[]
        for key,job in state['jobs'].items():
            result=job.get('result') or {}
            if (key=='verify-current:'+op or key.startswith('queued-proposal:'+op+':')) and result.get('result'):
                raw=Path(result['result']).read_bytes();formal=json.loads(raw)
                rows.append(dict(job=key,source=result['source'],source_sha256=result['source_sha256'],
                    result=result['result'],result_sha256=hashlib.sha256(raw).hexdigest(),
                    status=formal.get('status'),score=formal.get('geo_mean'),
                    timing=[{k:r.get(k) for k in ['uuid','latency_ms','reference_latency_ms']} for r in formal.get('per_workload',[]) if r.get('phase')=='timing']))
        context=dict(schema='hgkda-threeway-search-context-v1',status='formal_results_available',
            spec=dict(operator=op,source=str(source),source_sha256=best['source_sha256'],case=case),
            source=dict(path=str(source),sha256=best['source_sha256'],text=source.read_text(encoding='utf8')),
            execution_guidance=dict(current_batch_measurements=rows,
                hypotheses=(['连续flat大gather比六次独立stride3读写快；直接stride3方案已完整通过但明显更慢，勿再当已有效改法。',
                             '优先检查真实IR和设备路径，再尝试连续搬入UB后的局部重排/分块。现有结果尚不足以确认是哪条指令限制。']
                            if op=='cross_out' else
                            ['默认参数快速分派已略有改善；大输入固定40个program的4096块循环完整通过但更慢，不再重复该组合。',
                             '小中输入候选约17us、参考约14us；大输入差距更大。进一步降低分派或改善读写组织，必须检查两类输入。']),
                request='先提交完整可运行候选，再针对最差case做有界profiling；不要用目标PyTorch/aclnn算子替代Triton候选。'),
            limits=['这是本批原KGS完整调用数据；无新msprof指标或IR时不得宣称已确认设备瓶颈。',
                    '各结果与输入和源码SHA绑定；基于相同case的实际延迟评估修改，不只比较跨批汇总分数。',
                    '保留原容差、out/原地、stride、广播、NaN/Inf、当前stream；不读取评测文件或缓存张量值。'])
        out=batch/'analysis'/op/'agent-context-v3.json'
        out.parent.mkdir(parents=True,exist_ok=True)
        raw=(json.dumps(context,ensure_ascii=False,indent=2)+'\n').encode('utf8')
        if out.exists() and out.read_bytes()!=raw:raise FileExistsError(out)
        out.write_bytes(raw)
        print(op,len(rows),hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
