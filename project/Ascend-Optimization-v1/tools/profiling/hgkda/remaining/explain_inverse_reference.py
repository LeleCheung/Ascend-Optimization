"""用冻结的实际值证明重复行与逆矩阵残差；不改合同、不产生性能成绩。"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    probe_path=BATCH.parent/'next25-20261029-v2/analysis/linalg_inv_ex/contract-probe-fc75b4022866.json'
    probe=json.loads(probe_path.read_text(encoding='utf8'));case=probe['cases'][1]
    a=[[Fraction(x) for x in row] for row in case['input']['values']]
    ref=case['outputs'][0]['npu_reference'];x=[[Fraction(v) for v in row] for row in ref['values']]
    product=[[sum(a[i][k]*x[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    residual=[[float(product[i][j]-int(i==j)) for j in range(3)] for i in range(3)]
    # 输入两行逐位相同，其与任何矩阵相乘的两行也相同，不能等于单位阵。
    repeated=case['input']['values'][0]==case['input']['values'][2]
    assert repeated and product[0]==product[2]
    result=dict(status='observed_reference_semantic_conflict',probe=str(probe_path),
        probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),case=case['case'],
        identical_input_rows=[0,2],exact_rank_deficient=True,npu_reference_info=case['outputs'][1]['npu_reference']['values'],
        cpu_observation_info=case['cpu_observation']['info']['values'],
        npu_cpu_output_values_identical=ref['values']==case['cpu_observation']['inverse']['values'],
        reference_inverse_max_absolute=max(abs(v) for row in ref['values'] for v in row),
        exact_arithmetic_ax_minus_identity=residual,max_absolute_residual=max(abs(v) for row in residual for v in row),
        scope='使用已采集二进制浮点值转精确有理数作离线核验；不是新设备采集或KGS结果',
        conclusion='冻结输入具有完全重复行，数学上不可逆；本机参考返回有限大值和info=0。候选正确识别奇异仍会违反现有逐值合同。CPU/NPU观察值相同不单独证明CPU回退，实际路径另查trace。原合同和失败继续保留。')
    out=BATCH/'analysis/linalg_inv_ex/reference-mathematical-conflict.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
