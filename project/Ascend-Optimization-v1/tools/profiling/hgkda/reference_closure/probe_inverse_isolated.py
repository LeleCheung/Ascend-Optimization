"""CANN inverse每个冻结用例独立进程，避免一个设备异常污染后续用例。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def child(args):
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    case=read(args.case);source=Path(args.source);leaf=Path(args.leaf);output=Path(args.output)
    row=dict(case=case['name'],spec=case['inputs']['spec'],source_sha256=sha(source),physical_device=7,
        status='started',info_supported=False,reference_kind='aclnnInverse_only_no_info')
    save(output,row)
    try:
        oracle=module('isolated_inverse_oracle',leaf/'oracle.py');candidate=module('isolated_inverse_cann',source)
        inputs=oracle.gen_inputs(case,'npu:0');reference_inputs=oracle.gen_inputs(case,'npu:0')
        matrix=inputs['A'].cpu();raw=matrix.contiguous().numpy().tobytes()
        row['input_sha256']=hashlib.sha256(raw).hexdigest();row['shape']=list(matrix.shape)
        reference,reference_info=oracle.run(**reference_inputs);torch.npu.synchronize()
        row['reference_info']=reference_info.cpu().tolist();row['reference_finite']=bool(torch.isfinite(reference.cpu()).all())
        save(output,row)
        # CANN inverse不提供info；不能借用oracle的info拼出inv_ex结果。
        actual=candidate.run(**inputs);torch.npu.synchronize();value=actual.cpu();expected=reference.cpu()
        row['finite']=bool(torch.isfinite(value).all())
        row['inverse_matched']=bool(torch.allclose(value,expected,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True))
        residual=matrix.double()@value.double()-torch.eye(matrix.shape[-1],dtype=torch.float64)
        row['residual_max']=float(residual.abs().max()) if torch.isfinite(residual).all() else None
        row['max_absolute_difference']=float((value-expected).abs().max()) if torch.isfinite(value-expected).all() else None
        row.update(status='inverse_values_passed' if row['inverse_matched'] else 'inverse_values_mismatch')
    except Exception as e:row.update(status='execution_failed',error=type(e).__name__+': '+str(e)[:6000])
    save(output,row)
    print(json.dumps(row,ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ['source','leaf','output']:parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--case',type=Path);parser.add_argument('--case-timeout',type=int,default=180);args=parser.parse_args()
    if args.case:return child(args)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    out=args.output;out.mkdir(parents=True,exist_ok=True);rows=[]
    for phase in ['correctness','timing']:
        cases=[json.loads(line) for line in (args.leaf/(phase+'.jsonl')).read_text(encoding='utf8').splitlines() if line.strip()]
        for case in cases:
            directory=out/case['name'];directory.mkdir(exist_ok=True);spec=directory/'case.json';result=directory/'result.json';receipt=directory/'process.json'
            if spec.exists():assert read(spec)==case
            else:save(spec,case)
            if not receipt.exists():
                if result.exists():raise RuntimeError('子进程已有结果但未确认退出，不重复运行 '+str(result))
                command=[sys.executable,str(Path(__file__).resolve()),'--source',str(args.source),'--leaf',str(args.leaf),'--output',str(result),'--case',str(spec)]
                started=time.time();timed_out=False
                with (directory/'process.log').open('wb') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    save(directory/'started.json',dict(pid=process.pid,command=command,started_epoch=started,source_sha256=sha(args.source)))
                    try:code=process.wait(timeout=args.case_timeout)
                    except subprocess.TimeoutExpired:
                        timed_out=True;process.kill();code=process.wait(timeout=30)
                save(receipt,dict(returncode=code,timed_out=timed_out,elapsed_seconds=time.time()-started,command=command))
            value=read(result) if result.exists() else dict(case=case['name'],status='process_failed_without_result')
            value['phase']=phase;value['process']=read(receipt);value['process_log_sha256']=sha(directory/'process.log');rows.append(value)
    result=dict(status='partial_inverse_only_audit',eligible_full_inv_ex_contract=False,info_supported=False,
        source=str(args.source),source_sha256=sha(args.source),probe_sha256=sha(__file__),physical_device=7,
        contract_sha256={n:sha(args.leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
        inverse_values_passed=sum(r['status']=='inverse_values_passed' for r in rows),total=len(rows),cases=rows,
        scope='每case独立进程；计时输入只核验数值和残差，不生成正式性能分数；无info不视为inv_ex完整参考')
    save(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__':main()
