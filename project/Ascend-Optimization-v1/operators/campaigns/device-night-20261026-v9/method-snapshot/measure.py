"""一次profiling内交错两个单kernel函数，保留原始task_time并独立核对计时。"""
import csv
import hashlib
import json
import math
from pathlib import Path


def pair(functions,symbols,out):
    from triton.backends.ascend.testing import do_bench_npu_profiler
    keys=list(functions)
    if len(keys)!=2:raise ValueError('只支持两个已捕获的单kernel提交')
    raw=do_bench_npu_profiler([functions[k] for k in keys],warmup=2,active=6,prof_dir=str(out),keep_res=True)
    files=list(out.rglob('task_time*.csv'))
    if len(files)!=1:raise ValueError('设备task_time不唯一')
    with files[0].open(encoding='utf-8-sig',newline='') as f:allrows=list(csv.DictReader(f))
    rows=allrows[1:-1]
    if len(rows)!=16:raise ValueError('应为2函数×8次单kernel，实际'+str(len(rows)))
    values={};details=[]
    for i,key in enumerate(keys):
        tasks=rows[i*8:(i+1)*8]
        if any(symbols[key] not in r.get('kernel_name','') for r in tasks):raise ValueError('设备任务与捕获kernel身份不一致')
        durations=[float(r['task_time(us)']) for r in tasks[2:]]
        if any(not math.isfinite(v) or v<=0 for v in durations):raise ValueError('无效设备任务时长')
        number=sum(durations)/6/1000
        if not math.isclose(number,float(raw[i]),rel_tol=1e-9,abs_tol=1e-12):raise ValueError('独立重算与后端计时不同')
        values[key]=number;details.append(dict(key=key,symbol=symbols[key],active_device_us=durations,device_ms=number))
    metadata=dict(scope='真实设备单kernel任务时间；2次warmup＋6次active；两函数同次profiling，轮间反转顺序',
        csv_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),csv_path=str(files[0]),records=details)
    (out/'measurement.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return values
