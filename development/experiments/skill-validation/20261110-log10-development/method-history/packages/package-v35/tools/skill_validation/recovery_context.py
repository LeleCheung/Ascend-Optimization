"""将当前源码的启动诊断及完整性能反馈提供给模型，拒绝借用其他源码的采集。"""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from hgkda.artifacts import unwrap
from hgkda.context import knowledge


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(config):
    run=Path(config['run_directory']);workspace=run/'workspace'
    source=workspace/'starter.py';digest=sha(source)
    probe=Path(config['probe_directory']);measured=read(probe/'probe.json')
    feedback=unwrap(read(workspace/'performance-feedback.json'))
    assert measured['source_sha256']==feedback['source_sha256']==digest
    actual_ir=workspace/'actual-ir';actual_ir.mkdir(exist_ok=True)
    artifacts=[]
    for index,entry in enumerate(measured['compiled']):
        for item in entry['artifacts']:
            path=Path(item['path']);assert sha(path)==item['sha256']
            if path.suffix=='.npubin':continue
            target=actual_ir/(str(index)+'-'+path.name);shutil.copyfile(path,target)
            artifacts.append(dict(path=str(target),sha256=item['sha256'],kernel=entry['kernel']))
    # 轨迹单列设备任务；不重复计算Computing等派生区间。
    trace_root=Path(config['trace_directory']);trace_tasks=[]
    assert read(trace_root/'probe.json')['source_sha256']==digest, '任务轨迹与当前源码不同'
    for item in read(trace_root/'traces.json'):
        path=Path(item['path']);assert sha(path)==item['sha256']
        document=read(path)
        events=document['traceEvents'] if isinstance(document,dict) else document
        tasks=[e for e in events if e.get('ph')=='X' and e.get('args',{}).get('Task Type')]
        assert tasks, '轨迹缺少厂商设备任务记录'
        trace_tasks.append(dict(case=item['case'],side=item['side'],trace_sha256=item['sha256'],
            kernel_names=sorted({e['name'] for e in tasks}),count=len(tasks),
            durations_us=[e['dur'] for e in tasks],median_us=statistics.median(e['dur'] for e in tasks)))
    library=knowledge('selu_',hypotheses=True)
    brief=dict(operator='selu_',source_sha256=digest,formal_feedback=feedback,
        independent_launch_probe=measured['measurements'],device_tasks=trace_tasks,
        launchers=[{k:v for k,v in entry.items() if k not in ['artifacts','metadata']} for entry in measured['compiled'][:2]],
        mechanisms=library['mechanisms'],knowledge_retrieval=library['retrieval'],actual_ir=artifacts,
        limitations=['异步提交包含宿主逻辑及可能背压，不以wall-submit推断设备耗时。',
            '设备任务来自独立profiling窗口；不得代替KGS完整调用成绩。',
            '本轮不复用旧SHA的PipeUtilization或Roofline，不虚构缺失组件指标。',
            '历史启动优化与当前诊断提供待验证方法，组合收益由新源码完整评测确认。'])
    (workspace/'context-brief.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return brief
