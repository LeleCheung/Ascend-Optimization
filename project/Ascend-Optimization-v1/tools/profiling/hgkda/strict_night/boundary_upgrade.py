"""在下一个尚未启动的KG工作区边界加载控制器更新，不中断当前搜索。"""
import hashlib
import importlib.util
from pathlib import Path
import sys


def upgrade(runner,op,attempt,seed,overlay):
    project=runner.q.project
    source=project/'tools/profiling/hgkda/strict_night/run.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if getattr(runner,'_strict_controller_sha256',None)==digest:return
    number=int(attempt.rsplit('-',1)[-1])
    workspace=runner.q.runtime/'kg-runs'/(op+'-'+runner.batch.name+'-'+attempt)
    assert not workspace.exists(),'不对运行中的搜索做控制器切换'
    # 当前调用在前一项已结束、服务空闲后到达；先再次确认，不处理未知活动任务。
    runner.wait()
    spec=importlib.util.spec_from_file_location('strict_controller_'+digest[:12],source)
    updated=importlib.util.module_from_spec(spec);spec.loader.exec_module(updated)
    before=type(runner).__module__
    methods={}
    for file in sorted(source.parent.glob('*.py')):
        raw=file.read_bytes();identity=hashlib.sha256(raw).hexdigest()
        frozen=runner.batch/'method-snapshots/by-sha'/identity/file.name
        frozen.parent.mkdir(parents=True,exist_ok=True)
        if frozen.exists():assert frozen.read_bytes()==raw
        else:frozen.write_bytes(raw)
        methods[file.name]=dict(path=str(frozen),sha256=identity)
    record=dict(from_module=before,controller_source=str(source),controller_sha256=digest,
                workspace_not_started=True,methods=methods,status='loaded_at_idle_boundary',
                operation='仅在串行空闲边界加载控制器；上一项工作区保持原件')
    record_path=runner.batch/'analysis'/op/('controller-boundary-'+str(number)+'.json')
    updated.save(record_path,record)
    runner.__class__=updated.Runner
    runner._strict_controller_sha256=digest
    sys.path.insert(0,str(source.parent))
    # 只在串行边界替换检查模块；旧调用已返回，没有正在运行的ledger。
    sys.modules.pop('measurements',None)
    runner.refresh_existing_scores()
    runner.recover_meaningful_selection()
    runner.process_proposals()
    runner.verify_reviewed_best()
    runner.persist()
    record['status']='previous_candidates_finalized'
    updated.save(record_path,record)
    runner.prepare_paired(op,seed)
    # 刚由旧控制器准备、尚未加载的覆盖层也补齐同内容评测缓存。
    adapter=overlay/'kernelgen/tools/kernelgen_server_adapter.py'
    text=adapter.read_text()
    if '_strict_night_evaluate = evaluate_bundle' not in text:
        adapter.write_text(text+'''

_strict_night_evaluate = evaluate_bundle
def evaluate_bundle(bundle, context, *, run_control=None):
    from kernelgen.tools.hgkda_strict_admission import cached_evaluate
    return cached_evaluate(_strict_night_evaluate, bundle, context,
                           _settings_from_context(context), run_control)
''')
    runner.refresh_context(op,number,seed,overlay)
