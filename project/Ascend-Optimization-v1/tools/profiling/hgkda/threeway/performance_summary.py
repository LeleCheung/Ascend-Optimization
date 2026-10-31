"""汇总同源码、同合同的独立完整结果，保留未过线测量，避免选中最高复测值。"""
import math


def aggregate_independent(records, source_sha256):
    unique={}
    for row in records:
        if row.get('status')!='qualified' or row.get('source_sha256')!=source_sha256:continue
        path=row['result'];score=row['score']
        if not isinstance(score,(int,float)) or not math.isfinite(score) or score<=0:
            raise ValueError('完整成绩不是正有限数')
        if path in unique and unique[path]!=score:raise ValueError('同一结果路径出现不同成绩')
        unique[path]=score
    if not unique:return {}
    return dict(display_score=math.exp(sum(math.log(score) for score in unique.values())/len(unique)),
                stable_above_threshold=all(score>.8 for score in unique.values()),
                independent_measurements=len(unique),independent_results=list(unique))
