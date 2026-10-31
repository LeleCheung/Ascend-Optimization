"""新目标的同输入原生库成对采集；真实路径先发现，多kernel逐项保留。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(args,log,timeout=300,env=None):
    with log.open('w') as stream:r=subprocess.run([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,env=env)
    if r.returncode:raise RuntimeError('采集失败，见 '+str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--simulator',action='store_true');a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    spec=json.loads(a.spec.read_text(encoding='utf8'));state=dict(status='running',spec=spec,sides={},
        tools_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')});save(a.output/'state.json',state)
    qualification=a.output/'reference-qualification.json'
    library=Path('/data/hanle/ascend-optimization/runtime/next25-ascendc-build/lib/libhgkda_negative.so')
    if spec['operator']=='negative_' and library.is_file():
        try:
            run([sys.executable,HERE/'check_references.py','--output',qualification,'--library',library],a.output/'reference-qualification.log')
            state['source_built_ascendc_qualification']=str(qualification)
        except Exception as error:state['source_built_ascendc_qualification_error']=str(error)
    sides=['candidate','pytorch']+(['cann'] if spec['operator'] in ['expm1_','native_group_norm'] else [])
    if spec['operator']=='negative_' and spec['case']['inputs']['spec']['dtype']=='float32' and library.is_file():
        sides.append('ascendc')
    for side in sides:
        out=a.output/side;out.mkdir();args=[sys.executable,HERE/'driver.py','--spec',a.spec,'--side',side]
        row=dict(status='running');state['sides'][side]=row
        try:
            run(args+['--semantic',out/'semantics.json'],out/'semantic.log')
            run(args+['--trace',out/'trace.json'],out/'trace.log')
            data=json.loads((out/'trace.json').read_text(encoding='utf8'));events=data.get('traceEvents',[]) if isinstance(data,dict) else data
            device=[e for e in events if e.get('ph')=='X' and ('kernel' in str(e.get('cat','')).lower() or 'Task Type' in e.get('args',{}) or 'memcpy' in e.get('name','').lower())]
            names=list(dict.fromkeys(e['name'] for e in device if e.get('name')))
            row.update(device_events=device,selected=names)
            if not names:raise RuntimeError('没有发现可核验设备kernel，保留trace')
            captures=[]
            # torch_npu trace可显示aclnn调度别名，msprof op需要实际二进制kernel名。
            # 对单kernel路径仅在唯一漏选符号时建立映射；多kernel不按名称相似性猜测。
            app=shlex.join([str(x) for x in args])
            discovery=out/'op-discovery.log'
            run(['msprof','op','--application='+app,'--kernel-name=__hgkda_discovery_no_match__','--launch-count=1','--warm-up=0','--aic-metrics=Roofline','--output='+str(out/'op-discovery')],discovery,timeout=180)
            discovered=list(dict.fromkeys(re.findall(r'Kernel (.+?) skipped: not selected via --kernel-name\.',discovery.read_text(errors='replace'))))
            if set(names).issubset(discovered):
                actual_names=names
            elif len(names)==1 and len(discovered)==1:
                actual_names=discovered
            else:
                raise RuntimeError('trace别名与实际kernel需核对；拒绝多kernel盲目映射 '+repr(dict(trace=names,op=discovered)))
            row['kernel_name_binding']=dict(trace_names=names,op_names=actual_names,discovery_log=str(discovery),method='精确符号交集，或唯一实际符号与唯一目标任务对应')
            names=actual_names
            row['selected_op_symbols']=names
            for i,name in enumerate(names):
                profile=out/('kernel-%03d'%i)
                app=shlex.join([str(x) for x in args])
                cmd=['msprof','op','--application='+app,'--kernel-name='+name,'--launch-count=1','--warm-up=0','--aic-metrics=Roofline','--output='+str(profile)]
                run(cmd,out/('kernel-%03d.log'%i),timeout=480)
                basics=list(profile.rglob('OpBasicInfo.csv'))
                assert len(basics)==1,'基本信息不唯一'
                with basics[0].open(encoding='utf-8-sig',newline='') as f:basic=list(csv.DictReader(f))
                assert len(basic)==1 and basic[0]['Op Name']==name,'设备路径与OpProf目标不匹配'
                captures.append(dict(kernel=name,directory=str(profile),basic=basic[0],files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in profile.rglob('*.csv')}))
            row.update(status='completed',captures=captures)
            if side=='candidate':
                run(args+['--ir',out/'ir'],out/'ir.log')
                if a.simulator and len(names)==1:
                    raw=out/'simulator'
                    cache=Path('/data/hanle/ascend-optimization/runtime/next25-compiler-cache')/spec['operator']/hashlib.sha256(str(out).encode()).hexdigest()[:16]
                    cache.mkdir(parents=True,exist_ok=True);row['simulator_compiler_cache']=str(cache)
                    env=os.environ.copy();env.update(HGKDA_SIMULATOR='1',TRITON_DISABLE_LINE_INFO='false',TRITON_CACHE_DIR=str(cache))
                    env['LD_LIBRARY_PATH']='/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:'+env.get('LD_LIBRARY_PATH','')
                    # 使用小型显式规格；调用方须确认输入仍走待研究路径，不能静默缩形状。
                    command=['msprof','op','simulator','--application='+app,'--kernel-name='+names[0],'--soc-version=Ascend910B4','--core-id=0','--dump=on','--launch-count=1','--timeout=3','--output='+str(raw/'kernel-000')]
                    try:run(command,out/'simulator.log',timeout=240,env=env)
                    except Exception as e:row['simulator_error']=str(e)
                    from kernelgen_server.profiling.ascend.ascend_instruction import build_instruction_report
                    listing,warnings=build_instruction_report(raw,names,'Ascend910B4',{str(Path(spec['source'])):Path(spec['source']).read_text(encoding='utf8')})
                    save(out/'listing.json',listing);row['simulator_warnings']=warnings
                    row['simulator_status']='unavailable' if not listing.get('kernels') else 'partial' if row.get('simulator_error') or re.search(r'Child process exited with status [1-9]',(out/'simulator.log').read_text()) else 'completed'
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
        save(a.output/'state.json',state)
    state['status']='completed' if all(r['status']=='completed' for r in state['sides'].values()) else 'partial'
    save(a.output/'state.json',state)
    print(json.dumps(dict(status=state['status'],sides={k:v['status'] for k,v in state['sides'].items()})),flush=True)


if __name__=='__main__':main()
