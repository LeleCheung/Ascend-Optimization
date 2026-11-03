"""复用已验证的KG严格Triton覆盖层，在独立新批搜索；不改旧批或共享服务。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'
OLD=PROJECT/'operators/campaigns/strict-night-20261102'
ENGINE=HERE.parent/'strict_night'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--operator',required=True,choices=['expm1_','cumulative_trapezoid'])
    parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args()
    assert a.attempt in [1,2]
    assert (BATCH/'controller-process.json').exists(),'必须在已持项目锁的新批队列内运行'
    sys.path.insert(0,str(ENGINE))
    spec=importlib.util.spec_from_file_location('strict_five_search_base',ENGINE/'run.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    base.BATCH=BATCH
    # 参数化本次调用的状态文件，旧方法及旧批状态完全不改。
    class Search(base.Runner):
        def __init__(self):
            super().__init__();self.path=BATCH/'search-state.json'
            self.state=read(self.path) if self.path.exists() else dict(status='created',jobs={},operators={},started_at=self.module.now())
        def process_proposals(self):pass
        def verify_reviewed_best(self):pass
        def prepare_paired(self,op,seed):pass
        def report(self):
            save(BATCH/'search-progress.json',dict(status=self.state['status'],current_job=self.state.get('current_job'),
                best={op:{k:r.get('best',{}).get(k) for k in ['source','source_sha256','search_score','qualification']}
                      for op,r in self.state['operators'].items()},
                frozen_accepted=25,scope='新搜索中候选不自动计达标；人工审查、完整补测和合格原生参考仍需独立核验。'))
        def context(self,op,seed):
            value=super().context(op,seed)
            prior=read(OLD/'state.json')['operators'][op]
            history=[]
            for attempt in prior.get('attempts',[]):
                for row in attempt.get('candidates',[]):
                    item={k:row.get(k) for k in ['source','source_sha256','search_score','qualification','search_result']}
                    item['extras']=row.get('extras');history.append(item)
            for row in prior.get('proposals',[]):
                history.append({k:row.get(k) for k in ['source','source_sha256','search_score','qualification','search_result','extras']})
            value['frozen_previous_attempts']=history
            if op=='expm1_':
                paired=BATCH/'analysis/expm1_/paired-start-v1/summary.json'
                if paired.exists():
                    data=read(paired)
                    value['new_same_input_profile']=dict(path=str(paired),sha256=sha(paired),
                        status=data['status'],comparison=data.get('comparison'),comparison_error=data.get('comparison_error'),
                        sides={name:{k:r.get(k) for k in ['status','kernels','error']} for name,r in data['sides'].items()})
                value['diagnosis']+=('上一轮已真实完成两次4稿搜索及完整块/启动桥修正；8192整块v5正式30/30、63/63、整块12/12，'
                    '但0.514×低于0.561×当前最佳。不要重复相同分块与公式。新配对指标显示候选Vector主要FP32活动、UB存在冲突；'
                    '优先核对reference与candidate指标，研究公式中重复比较/选择、跨块FP32临时缓冲、编译实际UB与流水。'
                    '保持小值相对误差、负零、NaN/Inf、原地一次执行及stride/stream语义。'
                    '大输入公式与小输入均须通用，不允许按随机输入/seed/用例名特化。'
                    '本轮首稿可复用种子已有成绩，不要重复完整评测同一实现；每份实质改动一次KGS。')
                value['previous_exact_v5']=dict(path=str(OLD/'proposals/expm1_/codex-exact-zero-v5'),
                    source_sha256='ab6d7aeab424f06e2de4860f2aed3b807a457eeb9784c95d4955a81d99b1f56c',
                    formal='30/30',supplement='63/63',exact='12/12',PyTorch=0.5141239825103185)
            return value
    # 新合同与旧合同均为原PyTorch，种子沿用25项交付中的最佳。
    record=next(r for r in read(BATCH/'plan.json')['records'] if r['operator']==a.operator)
    leaf=BATCH/record['catalog_leaf'];original=BATCH/'original-catalog/ops'/a.operator
    if not original.exists():
        import shutil
        shutil.copytree(leaf,original,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
        assert (original/name).read_bytes()==(leaf/name).read_bytes()
    runner=Search()
    assert not runner.state.get('current_job'),'已有搜索未确认终态，保留现场'
    # 旧KG正式源码的完整测量缓存逐字节复用，避免把种子或历史新稿重复计时挑分。
    old_cache=OLD/'evaluation-cache';cache=BATCH/'evaluation-cache'
    if not cache.exists():
        import shutil
        shutil.copytree(old_cache,cache,ignore=shutil.ignore_patterns('__pycache__'))
    try:
        runner.search(a.operator,a.attempt)
        runner.state.update(status='search_completed_pending_review',current_job=None,finished_at=runner.module.now());runner.persist()
    except Exception as error:
        runner.state.update(status='requires_review',error=type(error).__name__+': '+str(error));runner.persist();raise
    out=BATCH/'analysis'/a.operator/('search-'+str(a.attempt)+'-receipt.json')
    save(out,dict(status='completed',operator=a.operator,attempt=a.attempt,
        search_state=str(runner.path),search_state_sha256=sha(runner.path),frozen_accepted=25,
        attribution='KG/HG-KDA＋CC/DeepSeek生成；Codex分析新配对指标、编排和审查。新稿不自动计达标。'))


if __name__=='__main__':main()
