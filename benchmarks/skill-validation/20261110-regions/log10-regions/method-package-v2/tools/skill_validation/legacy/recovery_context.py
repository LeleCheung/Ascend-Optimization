"""兼容早期SELU恢复合同；新实验使用transfer_context，不扩展此分支。"""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from hgkda.artifacts import unwrap
from hgkda.context import knowledge


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_initial(config):
    """仅重放早期SELU上下文格式；不准备新候选或扩展到其他算子。"""
    assert config['job']['operator']=='selu_', '旧上下文仅适用于SELU；新实验使用operator_transfer配置'
    run=Path(config['run_directory']);workspace=run/'workspace'
    comparison=workspace/'comparison.json';context=workspace/'context.json'
    if not comparison.exists():
        subprocess.run([sys.executable,'-m','hgkda','compare','--directory',config['paired_directory'],
                        '--output',str(comparison)],check=True,stdout=subprocess.DEVNULL)
    if not context.exists():
        subprocess.run([sys.executable,'-m','hgkda','context','--operator','selu_',
            '--source',str(workspace/'starter.py'),'--comparison',str(comparison),
            '--ir-manifest',str(run/'baseline-ir/manifest.json'),'--output',str(context)],
            check=True,stdout=subprocess.DEVNULL)
    data=unwrap(read(context));paired=data['facts']['paired']
    local_ir=workspace/'actual-ir';local_ir.mkdir(exist_ok=True)
    for item in data['facts']['compiled_ir']:
        shutil.copyfile(item['path'],local_ir/Path(item['path']).name)
    selected=[row for row in paired['differences'] if any(k in row['metric'] for k in
                ['ratio','time','instructions','datas'])]
    selected.sort(key=lambda row:abs((row.get('candidate_over_reference') or 1)-1),reverse=True)
    brief=dict(source_sha256=data['source']['sha256'],case=data['case'],reference_kind=data['reference_kind'],
        differences=selected[:12],transfer_granularity=paired['derived_differences'],hypotheses=data['hypotheses'],
        mechanisms=data['knowledge']['mechanisms'],full_context=str(context),
        measurement_limits=paired['limitations'],
        interpretation_rules=['采集为零不自动等于低于分辨率或没有开销；保留未知原因。',
            '分块payload不等于实际每条搬运指令粒度，须从编译产物或计数核验。',
            '不得以全任务字节数除以逐block平均时间建立Roofline或声称达到带宽上限。',
            '实际类型精度是否改变须检查新候选IR；Python源码表达式不能替代编译证据。'],
        actual_ir=[str(local_ir/Path(row['path']).name) for row in data['facts']['compiled_ir']],
        reference='CANN原生；正式性能只由KGS完整评测确定')
    (workspace/'context-brief.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return brief


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
