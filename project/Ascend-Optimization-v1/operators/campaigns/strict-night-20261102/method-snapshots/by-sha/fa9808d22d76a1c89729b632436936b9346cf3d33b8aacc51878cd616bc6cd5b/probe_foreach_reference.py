"""原生CANN ForeachExpm1资格诊断；仅在项目串行空闲边界执行，不产生候选成绩。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_build(path):
    build=json.loads(Path(path).read_text(encoding='utf8'))
    assert build['status']=='built'
    for name in ['source','binary']:
        assert sha(build[name])==build[name+'_sha256'],'参考适配文件改变：'+name
    for row in build['dependencies'].values():
        assert sha(row['frozen'])==row['sha256'],'冻结厂商头文件改变'
    return build


def process(runner):
    from night_common import BATCH,HERE,PY,read,save
    out=BATCH/'reference-probes/native-foreach-expm1-v1'
    build=out/'build.json';result=out/'numerics.json'
    if not build.exists():return
    verify_build(build)
    key='reference-probe:native-foreach-expm1-v1'
    previous=runner.state['jobs'].get(key,{})
    if previous.get('status')=='completed':
        assert result.exists() and read(result)['build_sha256']==sha(build)
        return
    if result.exists():
        value=read(result);assert value['build_sha256']==sha(build)
        frozen=out/('checker-'+value['checker_sha256']+'.py')
        assert sha(frozen)==value['checker_sha256']
    else:
        assert previous.get('status')!='running','上次探测无终态回执，不重发'
        runner.wait()
        original=runner.state.get('current_job')
        checker=HERE/'probe_foreach_reference.py'
        frozen=out/('checker-'+sha(checker)+'.py')
        if frozen.exists():assert frozen.read_bytes()==checker.read_bytes()
        else:frozen.write_bytes(checker.read_bytes())
        runner.state['current_job']=key
        runner.state['jobs'][key]=dict(status='running',build=str(build),build_sha256=sha(build),checker=str(frozen))
        runner.persist()
        runner.cmd(key,[PY,frozen,'--build',build,'--leaf',BATCH/'original-catalog/ops/expm1_','--output',result],900)
        runner.state['current_job']=original
    value=read(result)
    assert value['qualified_CANN'] is False and value['performance_measured'] is False
    runner.state['jobs'][key]=dict(status='completed',result=str(result),result_sha256=sha(result),
        outcome=value['status'],scope='原生参考数值探测，不新增CANN达标数，不修改合同')
    runner.persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['build','leaf','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'已有探测不覆盖或重发'
    build=verify_build(args.build)
    import torch
    import torch_npu  # noqa: F401
    assert torch.__version__==build['torch'] and torch._C._GLIBCXX_USE_CXX11_ABI==build['abi']
    spec=importlib.util.spec_from_file_location(build['module_name'],build['binary'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location('foreach_original_oracle',args.leaf/'oracle.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    cases=[json.loads(line) for line in (args.leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    rows=[]
    samples=[0.,-0.,1.e-8,-1.e-8,1.e-6,-1.e-6,1.e-4,-1.e-4,
             0.006,-0.006,0.0099,-0.0099,0.01,-0.01,0.0101,-0.0101,
             0.03,-0.03,0.0624,-0.0624,0.0625,-0.0625,float('nan'),float('inf'),-float('inf')]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        tolerance=next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
        cpu=torch.tensor(samples,dtype=dtype);expected_cpu=torch.expm1(cpu)
        for alias in [False,True]:
            for strided in [False,True]:
                row=dict(dtype=str(dtype),alias_output=alias,strided=strided,passed=False)
                try:
                    if strided:
                        base=torch.full((cpu.numel(),2),77,dtype=dtype,device='npu:0')
                        x=base[:,0];x.copy_(cpu);guard=base[:,1].clone()
                    else:x=cpu.to('npu:0')
                    before=x.clone();expected=oracle.run(A=x.clone())
                    actual=module.expm1(x,alias);torch.npu.synchronize();observed=actual.cpu()
                    row.update(input=[str(v) for v in cpu.tolist()],native=[str(v) for v in observed.tolist()],
                               cpu_reference=[str(v) for v in expected_cpu.tolist()])
                    torch.testing.assert_close(actual,expected,atol=tolerance['atol'],rtol=tolerance['rtol'],equal_nan=True)
                    row['original_NPU_tolerance_passed']=True
                    torch.testing.assert_close(observed,expected_cpu,atol=0,rtol=tolerance['rtol'],equal_nan=True)
                    row['small_relative_accuracy_passed']=True
                    zero=expected_cpu==0
                    assert torch.equal(torch.signbit(observed[zero]),torch.signbit(expected_cpu[zero])),'零符号不符合CPU数学参考'
                    if alias:assert torch._C._is_alias_of(actual,x),'原生别名路径没有返回输入存储'
                    else:torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                    if strided:torch.testing.assert_close(base[:,1],guard,atol=0,rtol=0,equal_nan=True)
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    value=dict(status='numerics_passed_pending_full_qualification' if all(r['passed'] for r in rows) else 'numerics_failed',
        api='aclnnForeachExpm1',list_length=1,passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,
        qualified_CANN=False,performance_measured=False,build=str(args.build),build_sha256=sha(args.build),
        checker_sha256=sha(__file__),oracle_sha256=sha(args.leaf/'oracle.py'),
        device=str(torch.npu.get_device_name(0)),
        scope='沿用expm1既有小值/零符号门槛；同时探测原生单张量list、输入输出别名与stride。不是完整合同或性能成绩。')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:value[k] for k in ['status','passed','total','qualified_CANN','performance_measured']}))


if __name__=='__main__':main()
