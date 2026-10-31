"""从原trace重算设备摘要，旧summary.json保留；无需重新使用NPU。"""
import json
from pathlib import Path
from trace_summary import summarize


def main():
    project=Path(__file__).resolve().parents[4]
    batch=project/'operators/campaigns/threeway-20261030'
    rows=[]
    for trace in sorted((batch/'paths').glob('*/*/*/trace.json')):
        spec=json.loads((trace.parent/'spec.json').read_text(encoding='utf8'))
        value=summarize(trace);value['spec']=spec
        out=trace.parent/'summary-v2.json'
        raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8')
        if out.exists() and out.read_bytes()!=raw:raise FileExistsError(out)
        out.write_bytes(raw)
        rows.append(dict(operator=spec['operator'],case=spec['case']['name'],
            implementation=spec['implementation_label'],source_sha256=spec['source_sha256'],
            summary=str(out),**{k:v for k,v in value.items() if k not in {'spec','device_tasks'}}))
    (batch/'path-index-v2.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(traces=len(rows),observed=sum(row['status']=='observed' for row in rows))))


if __name__=='__main__':main()
