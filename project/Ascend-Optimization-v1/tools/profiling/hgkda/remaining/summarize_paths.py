"""汇总绑定源码/输入的实际设备路径和IR；与正式完整调用成绩分开。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    rows=[]
    for summary in sorted((BATCH/'paths').glob('*/*/*/summary.json')):
        value=json.loads(summary.read_text(encoding='utf8'));spec=json.loads((summary.parent/'spec.json').read_text(encoding='utf8'))
        assert hashlib.sha256(Path(spec['source']).read_bytes()).hexdigest()==spec['source_sha256']
        assert hashlib.sha256(Path(value['trace']).read_bytes()).hexdigest()==value['trace_sha256']
        rows.append(dict(operator=spec['operator'],case=spec['case']['name'],spec=spec['case']['inputs']['spec'],
            source=spec['source'],source_sha256=spec['source_sha256'],label=spec['implementation_label'],
            status=value['status'],kernels=value['kernel_names'],device_task_count=len(value['device_tasks']),
            device_interval_union_us=value['device_interval_union_us'],summary=str(summary),trace=value['trace'],trace_sha256=value['trace_sha256']))
    ir=[]
    for manifest in sorted((BATCH/'actual-ir').glob('*/*/manifest.json')):
        value=json.loads(manifest.read_text(encoding='utf8'))
        ir.append(dict(operator=value['spec']['operator'],source_sha256=value['spec']['source_sha256'],manifest=str(manifest),
            entries=[dict(symbol=entry['symbol'],kinds=[item['kind'] for item in entry['artifacts']]) for entry in value['entries']]))
    index=dict(status='observed',rows=rows,actual_ir=ir,
        scope='同输入独立profiling单次设备任务区间并集；用于识别执行路径，不替代KGS完整调用、不与其分数平均')
    (BATCH/'path-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 实际设备路径', '', index['scope'], '', '|算子|用例|实现|任务数|设备区间并集/微秒|kernel|', '|---|---|---|---:|---:|---|']
    for row in rows:
        lines.append('|'+row['operator']+'|'+row['case']+'|'+row['label']+'|'+str(row['device_task_count'])+'|'+format(row['device_interval_union_us'],'.3f')+'|'+', '.join(row['kernels'])+'|')
    (BATCH/'设备路径摘要.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='observed',paths=len(rows),actual_ir=len(ir))))


if __name__=='__main__':main()
