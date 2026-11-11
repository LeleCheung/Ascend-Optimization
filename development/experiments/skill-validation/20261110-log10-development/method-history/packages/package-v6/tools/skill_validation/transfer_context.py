"""为多算子方法迁移整理当前源码、分类知识、合同与已绑定的实际采集。"""
import hashlib
import json
import math
from pathlib import Path
import shutil

from hgkda.artifacts import unwrap
from hgkda.context import knowledge


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(config):
    run=Path(config['run_directory']);workspace=run/'workspace'
    source=workspace/config.get('active_source','starter.py');digest=sha(source)
    operator=config['job']['operator'];library=knowledge(operator,hypotheses=True)
    # 模型只需可执行方法与来源摘要；完整历史路径和嵌套验证原件由宿主保管。
    visible=[{k:record[k] for k in ['id','operators','types','status','signal','mechanism','action','constraints',
              'match','application','source_sha256'] if k in record} for record in library['mechanisms']]
    for name in ['knowledge.json','contract.json','reference-qualification.json']:
        target=workspace/name
        if name=='knowledge.json':
            target.write_text(json.dumps(dict(operator_type=library['operator_type'],mechanisms=visible,
                retrieval=library['retrieval'],update_policy=library['update_policy']),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    compiled=run/'attempts'/digest/'compiled/verification.json'
    artifacts=[]
    if compiled.exists():
        record=read(compiled);assert record['source_sha256']==digest
        target_root=workspace/'actual-ir'/digest[:12];target_root.mkdir(parents=True,exist_ok=True)
        for index,entry in enumerate(record['entries']):
            for item in entry['artifacts']:
                if item['kind'] not in ['ttir','bcmlir']:continue
                path=Path(item['path']);assert sha(path)==item['sha256']
                target=target_root/(str(index)+'-'+path.name);shutil.copyfile(path,target)
                artifacts.append(dict(path=str(target.relative_to(workspace)),sha256=item['sha256'],kernel=entry['symbol']))
    preflight=None
    if (workspace/'latest-preflight.json').exists():
        row=read(workspace/'latest-preflight.json')
        if row.get('source_sha256')==digest:preflight=row
    feedback=None
    for name in ['next-performance-feedback.json','performance-feedback.json']:
        if (workspace/name).exists():
            feedback=unwrap(read(workspace/name));assert feedback['source_sha256']==digest
            break
    paired=None
    pointer=workspace/'paired-context-pointer.json'
    if pointer.exists():
        item=read(pointer)
        assert item['source_sha256']==digest,'成对采集不属于当前源码'
        context=Path(item['context']);assert sha(context)==item['sha256']
        full=unwrap(read(context));facts=full['facts']['paired']
        selected=[row for row in facts['differences'] if any(word in row['metric'] for word in
                  ['ratio','time','instructions','datas','conflict','wait'])]
        def deviation(row):
            ratio=row.get('candidate_over_reference')
            return abs(math.log(ratio)) if isinstance(ratio,(int,float)) and ratio>0 else 0
        selected.sort(key=deviation,reverse=True)
        paired=dict(case=full['case'],source_sha256=full['source']['sha256'],
            reference_kind=full['reference_kind'],differences=selected[:32],
            transfer_granularity=facts['derived_differences'],execution_groups=facts['execution_groups'],
            hypotheses=full['hypotheses'],limitations=facts['limitations'],
            full_context_sha256=item['sha256'],
            scope='保留变化最大的32项指标及完整搬运粒度；实际IR另列，不重复嵌入源码和历史知识')
    methods=read(Path(config['method_manifest']))
    for item in methods['files']:
        assert sha(Path(config['package'])/item['path'])==item['sha256'],item['path']
    runtime=workspace/'runtime-api.json'
    value=dict(operator=operator,phase=config['phase'],source_sha256=digest,active_source=source.name,
        contract=read(workspace/'contract.json'),knowledge_retrieval=library['retrieval'],
        mechanisms=visible,formal_feedback=feedback,preflight_feedback=preflight,actual_ir=artifacts,
        paired_evidence=paired,runtime_api=dict(path=runtime.name,sha256=sha(runtime)) if runtime.exists() else None,
        method_manifest_sha256=sha(Path(config['method_manifest'])),
        interpretation_rules=[
          '知识是方法及条件，不提供历史最佳源码；候选全部修改由CC／DeepSeek完成。',
          '原始FlagGems起点可能含目标设备不合规回退，初次采集缺失不代表已验证其Triton路径。',
          '正式性能只采用KGS完整合同；profiling或设备任务时间与完整调用分别记录。',
          '缺少实际IR、Roofline或组件指标时保留未知，不根据源码虚构测量。',
          '不得把全任务工作量除以逐核平均时间；计数与时间的聚合范围必须一致。',
          '预检通过是交接审查的边界，正式性能未达标须依据反馈继续生成候选。'])
    (workspace/'context-brief.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return value
