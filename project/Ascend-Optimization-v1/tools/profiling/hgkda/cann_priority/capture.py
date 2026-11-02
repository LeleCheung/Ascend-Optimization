"""对最终源码及CANN参考采集同输入设备路径；独立时间不替代KGS。"""
import math
from pathlib import Path
from common import BATCH, NEXT, PY, THREE, load, read, save, sha


def capture(r, op, source, reference, leaf, label):
    cases = [__import__('json').loads(s) for s in (leaf/'timing.jsonl').read_text().splitlines() if s.strip()]
    # 首个小输入及最大连续FP32输入，避免只展示最有利的一个case。
    fp32 = [c for c in cases if c['inputs']['spec'].get('dtype') == 'float32'] or cases
    largest = max(fp32, key=lambda c: math.prod(c['inputs']['spec'].get('shape', [1])))
    chosen = [cases[0], largest]
    parser = load('cann_priority_trace_parser', THREE/'trace_summary.py')
    records = []
    for case in chosen:
        for side, code in [('candidate', source), ('CANN', reference)]:
            out = BATCH/'paths'/op/label/case['name']/side
            out.mkdir(parents=True, exist_ok=True)
            payload = dict(operator=op, leaf=str(leaf), source=str(code), source_sha256=sha(code), case=case,
                           scope='同输入独立设备trace；不替代正式完整调用成绩')
            spec = out/'spec.json'
            if spec.exists():assert read(spec) == payload
            else:save(spec, payload)
            trace = out/'trace.json'
            if not trace.exists():r.cmd('trace-'+op+'-'+label+'-'+case['name']+'-'+side,
                [PY, NEXT/'driver.py', '--spec', spec, '--side', 'candidate', '--trace', trace], 600)
            value = parser.summarize(trace)
            save(out/'summary.json', value)
            records.append(dict(side=side, case=case['name'], source_sha256=sha(code), trace=str(trace),
                                summary=str(out/'summary.json'), summary_sha256=sha(out/'summary.json')))
    save(BATCH/'paths'/op/label/'index.json', records)
    return dict(status='captured', records=len(records), index=str(BATCH/'paths'/op/label/'index.json'))
