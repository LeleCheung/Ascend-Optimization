"""只读输出当前批次状态，不读取模型思考记录。"""
import json
from pathlib import Path

batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/threeway-20261030')
state=json.loads((batch/'next-state.json').read_text())
print(json.dumps(dict(status=state.get('status'),current_job=state.get('current_job'),error=state.get('error'),
    operators={op:dict(score=(row.get('best') or {}).get('geo_mean'),status=row.get('status'),
                       tests=(row.get('best') or {}).get('num_passed'),supplement=(row.get('best') or {}).get('supplementary_status'))
               for op,row in state.get('operators',{}).items()},
    failed_jobs={key:row.get('error') for key,row in state.get('jobs',{}).items() if row.get('status')=='failed'},
    references=state.get('references',{})),ensure_ascii=False))
