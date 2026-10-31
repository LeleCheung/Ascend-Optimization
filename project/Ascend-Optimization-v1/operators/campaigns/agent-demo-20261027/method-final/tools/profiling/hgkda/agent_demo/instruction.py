"""只采已确认候选符号的单核模拟器指令，复用KGS解析与HG-KDA热点筛选。"""
import argparse,hashlib,importlib.util,json,os,re,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];REPO=PROJECT.parents[1]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--side',choices=['seed','candidate'],default='seed');p.add_argument('--attempt',default='');a=p.parse_args();b=a.batch.resolve()
    out=b/(a.side+'-instruction'+('-'+a.attempt if a.attempt else ''));assert not out.exists();out.mkdir()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    # 小型代表输入仍进入与1024相同的双输出BK256路径，不用不同kernel冒充目标。
    capture=b/(a.side+'-profile')/'candidate-float16-512';spec=json.loads((capture/'spec.json').read_bytes());source=Path(spec['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256']
    discovery=json.loads((capture/'discovery.json').read_bytes());names=discovery['selected'];assert len(names)==1
    # 模拟器将可见设备重映射为逻辑0；只在本批独立驱动中接受该映射，不改真机门禁。
    driver=out/'simulator_driver.py';text=(HERE.parent/'roofline/matrix_driver.py').read_text(encoding='utf8')
    text=text.replace("HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]",'HERE=Path('+repr(str(HERE.parent/'roofline'))+');PROJECT=HERE.parents[3]')
    text=text.replace("assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'","os.environ['ASCEND_RT_VISIBLE_DEVICES']='0'  # 独立模拟器驱动，模拟器清理/重写环境，逻辑卡固定0")
    driver.write_text(text,encoding='utf8')
    app=shlex.join([sys.executable,str(driver),'--spec',str(capture/'spec.json'),'--output',str(out/'compiled')])
    command=['msprof','op','simulator','--application='+app,'--kernel-name='+names[0],
        '--soc-version=Ascend910B4','--core-id=0','--dump=on','--launch-count=1','--timeout=3','--output='+str(out/'raw/kernel-000')]
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:'+env.get('LD_LIBRARY_PATH','')
    env.update(TRITON_DISABLE_LINE_INFO='false',TRITON_CACHE_DIR=str(out/'compiler-cache'),HGKDA_SIMULATOR='1')
    save(out/'request.json',dict(command=command,source_sha256=spec['source_sha256'],shape=spec['shape'],dtype=spec['dtype'],kernel=names[0],
        simulator_soc='Ascend910B4',physical_device='7',core='0',driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),timing_scope='模拟器指令时间，不是正式KGS延迟',transport='KGS同一采集命令与解析器，本批直接调用以限制为单核代表输入'))
    try:
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=240)
        save(out/'execution.json',dict(returncode=result.returncode))
    except subprocess.TimeoutExpired:
        save(out/'execution.json',dict(status='timeout',seconds=240));raise
    from kernelgen_server.profiling.ascend.ascend_instruction import build_instruction_report
    sources={str(source):source.read_text(encoding='utf8')}
    for path in (out/'compiled').rglob('*.txt'):
        if path.name in ['ttir.txt','bcmlir.txt','source.txt']:sources[str(path)]=path.read_text(encoding='utf8')
    listing,warnings=build_instruction_report(out/'raw',names,'Ascend910B4',sources)
    save(out/'listing.json',listing)
    s=importlib.util.spec_from_file_location('demo_adapter',PROJECT/'tools/profiling/hgkda_adapter.py');adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
    hotspots=adapter.instruction_hotspots(listing,source.read_text(encoding='utf8'))
    count=sum(c['listed_instruction_count'] for k in listing['kernels'] for c in k['cores'])
    mapped=sum(c['mapped_instruction_count'] for k in listing['kernels'] for c in k['cores'])
    log=(out/'stdout.txt').read_text(errors='replace')
    child_codes=[int(x) for x in re.findall(r'Child process exited with status (\d+)',log)]
    child_failed=any(x!=0 for x in child_codes)
    evidence_valid=bool(count and hotspots['status']=='matched' and result.returncode==0)
    status=('partial' if child_failed else 'completed') if evidence_valid else 'failed'
    save(out/'summary.json',dict(status=status,evidence_valid=evidence_valid,child_exit_codes=child_codes,instructions=count,mapped_instructions=mapped,warnings=warnings,hotspots=hotspots,
        source_sha256=spec['source_sha256'],parser='KGS ascend_instruction.build_instruction_report',filter='HG-KDA instruction_hotspots'))
    print(json.dumps(dict(status=status,instructions=count,mapped=mapped)),flush=True)
    if not evidence_valid:raise SystemExit(1)

if __name__=='__main__':main()
