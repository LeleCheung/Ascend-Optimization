"""核对阶段输出与CPU源级模型，保留边界和原件身份。"""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BATCH=HERE.parents[1]
probe=BATCH/'numerical-probes/inverse-stages-v2/result.json'
model=HERE/'cpu-literal-v5/result.json'
read=lambda f:json.loads(f.read_text(encoding='utf8'))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
p,c=read(probe),read(model)
assert p['source_sha256']==c['source_sha256']
assert p['status']=='measured' and p['device_executed'] is True
assert c['device_executed'] is False
assert all(row['instrumentation_matches_original'] for row in p['cases'])
models={r['case']:r for r in c['rows']}
rows=[]
for actual in p['cases']:
    expected=models[actual['case']]
    left=np.asarray(actual['stages'][0],dtype=np.float32)
    right=np.asarray([r['work'] for r in expected['stages']],dtype=np.float32)
    assert left.shape==right.shape
    differences=np.argwhere(left!=right)
    rows.append(dict(case=actual['case'],instrumentation_matches_original=True,
        matrix_count=int(left.shape[0]),different_lu_elements=len(differences),
        max_lu_abs_error=float(np.max(np.abs(left-right))),
        first_differences=[dict(index=list(map(int,index)),device=float(left[tuple(index)]),
            cpu_model=float(right[tuple(index)])) for index in differences[:4]]))
scalar=p['scalar_rows']
examples=[]
for row in scalar:
    if row['corrected_reciprocal']!=row['expected_corrected_reciprocal']:
        if any(r['denominator']==row['b'] for r in examples):continue
        examples.append(dict(denominator=row['b'],device=row['corrected_reciprocal'],
                             cpu_model=row['expected_corrected_reciprocal']))
summary=dict(status='measured_stage_comparison',source_sha256=p['source_sha256'],
    path_base='campaign_root',
    inputs=dict(probe=dict(path=probe.relative_to(BATCH).as_posix(),sha256=sha(probe)),
                cpu_model=dict(path=model.relative_to(BATCH).as_posix(),sha256=sha(model))),
    checker_sha256=sha(Path(__file__)),cases=rows,scalar_samples=len(scalar),
    scalar_mismatches=p['scalar_operand_mismatches'],reciprocal_examples=examples,
    findings=[
        '四组插桩最终输出均与原稿逐位一致，LU阶段已偏离CPU逐句模型；更改三角求解组织的v6正式误差仍与v5相同。',
        '第一组3×3矩阵的LU上三角末列有两项明显偏差；下半列补偿更新外的where应保留这些元素。',
        '标量分母与向量分子混合时，补偿乘加和修正商未见差异；常数分子的倒数有384/1024项差异。'],
    next_experiment=[
        'v7先屏蔽乘加操作数，再直接更新column，避免where同时依赖旧column与补偿临时值。',
        'v7填充行分母设1、有效行使用主元，保持非splat向量表达；原合同和容差不变。'],
    limitations='尚未定位具体编译pass或ISA；CPU模型不是设备模拟器。两项表达式修改均待NPU验证，不能据此宣布修复或CANN达标。')
out=HERE/'stage-comparison-v2.json'
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=summary['status'],lu_differences=[r['different_lu_elements'] for r in rows],
                     scalar_mismatches=summary['scalar_mismatches'])))
