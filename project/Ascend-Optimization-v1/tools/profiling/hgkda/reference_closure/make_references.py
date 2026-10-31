"""生成绑定二进制身份的CANN参考；单算子与兼容组合分别记账。"""
import argparse
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',choices=['v1','v2','v3','v4'],default='v1');args=parser.parse_args()
    revision=args.revision;build_path=BATCH/('cann-build.json' if revision=='v1' else 'cann-build-'+revision+'.json')
    build=json.loads(build_path.read_text(encoding='utf8'))
    binary=Path(build['binary']);assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']
    prefix='import hashlib,importlib.util\nfrom pathlib import Path\nimport torch,torch_npu\n'
    prefix+='p=Path('+repr(str(binary))+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(build['binary_sha256'])+'\n'
    prefix+='s=importlib.util.spec_from_file_location('+repr(build['module_name'])+',p)\n_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
    prefix+='assert torch.__version__=='+repr(build['torch'])+' and torch._C._GLIBCXX_USE_CXX11_ABI=='+repr(build['abi'])+'\n'
    functions={
      'fix':('def run(self):\n    return _cpp.trunc(self)\n','CANN_native_aclnnTrunc','浮点向零舍入；完整接口'),
      'expm1_':('def run(A):\n    return _cpp.expm1_(A)\n','CANN_aclnn_with_explicit_BF16_copy','FP16/FP32原地；BF16采用库非原地结果加NPU复制'),
      'fake_quantize_per_tensor_affine_cachemask_out':('def run(input,scale,zero_point,quant_min,quant_max,*,out0,out1):\n    return _cpp.fake_quant(input,scale,zero_point,quant_min,quant_max,out0,out1)\n','CANN_aclnn_with_explicit_BF16_conversion','FP16/FP32直接；BF16转FP32后计算并写回，全部成本计入'),
      'batch_norm_with_update_functional':('def run(input,weight=None,bias=None,running_mean=None,running_var=None,momentum=0.1,eps=1e-5):\n    return _cpp.batch_norm(input,weight,bias,running_mean,running_var,momentum,eps)\n','CANN_aclnnBatchNorm_functional_wrapper','复制running状态后训练模式，六输出；需验证reserve及统计类型'),
      'linalg_inv_ex':('def run(A,*,check_errors=False):\n    return _cpp.inverse_only(A)\n','CANN_aclnnInverse_partial_inverse_only','只返回inverse，不伪造info；仅资格诊断，不进入原inv_ex正式合同'),
    }
    records={}
    for op,(body,kind,scope) in functions.items():
        if revision!='v1' and op not in (['fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional'] if revision=='v2' else ['batch_norm_with_update_functional']):continue
        if revision!='v1':
            kind='CANN_aclnn_FP32_compatibility_composition'
            scope='显式FP32计算兼容组合，含输入/输出转换；BatchNorm补齐4D、独立running状态与默认affine；不是原生低精度单算子'
        out=BATCH/'references'/op if revision=='v1' else BATCH/('references-'+revision)/op
        source=out/'main.py';text=prefix+body;out.mkdir(parents=True,exist_ok=True)
        if source.exists() and source.read_text(encoding='utf8')!=text:raise ValueError('参考已冻结 '+op)
        source.write_text(text,encoding='utf8')
        record=dict(operator=op,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),kind=kind,scope=scope,build=str(build_path),revision=revision,
          binary=str(binary),binary_sha256=build['binary_sha256'],eligible_full_contract=op!='linalg_inv_ex')
        save(out/'provenance.json',record);records[op]=record
    save(BATCH/('reference-index.json' if revision=='v1' else 'reference-index-'+revision+'.json'),records);print(json.dumps(dict(status='prepared',revision=revision,references=list(records))))
if __name__=='__main__':main()
