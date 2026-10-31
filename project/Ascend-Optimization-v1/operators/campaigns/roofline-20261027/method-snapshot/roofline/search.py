"""由Roofline和流水差异提出的受控候选，保留全部失败和计时原件。"""
import argparse,hashlib,importlib.util,json,os,statistics,sys
from pathlib import Path
from candidate_template import CONFIGS,generate

HERE=Path(__file__).resolve().parent
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configs',nargs='+',choices=[c['id'] for c in CONFIGS]);a=p.parse_args()
    a.output=a.output.resolve();a.parent=a.parent.resolve()
    configs=[c for c in CONFIGS if not a.configs or c['id'] in a.configs]
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7';a.output.mkdir(parents=True)
    # 先登记预算、作用域与预期，之后不依据正式完整测试再调参。
    plan=dict(configs=configs,shapes=[1024,2048],dtypes=['float16','bfloat16'],repetitions=2,
        parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        hypothesis='半精度父版HBM经验达成率低且Cube活跃低于PyTorch；双输出累加/缓冲和同步可能阻碍流水。以单个较宽输出块替代两个相邻块，保持或降低逻辑请求量。',
        predictions=['单输出宽块减少两次dot之间的调度和临时缓冲','A逻辑请求不增加','Cube活跃或设备时间改善'],
        source='Codex依据本批Roofline、源码和IR提出的受控变换，未声称模型自动发现')
    save(a.output/'plan.json',plan)
    import torch
    import torch_npu
    sys.path.insert(0,str(HERE.parent/'mechanism'));helper=load(HERE.parent/'mechanism/probe.py','roof_search_helper')
    measure=load(HERE.parent/'device_mechanism/measure.py','roof_search_measure')
    compiler=load(HERE.parent/'device_mechanism/probe.py','roof_search_compiler')
    parent=load(a.parent,'roof_search_parent');state=dict(status='running',cases=[]);save(a.output/'state.json',state)
    parent_text=a.parent.read_text(encoding='utf8')
    for config in configs:
        directory=a.output/config['id'];directory.mkdir();source=directory/'candidate.py';source.write_text(generate(parent_text,config),encoding='utf8',newline='\n')
        row=dict(case=config['id'],config=config,status='running',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),measurements=[]);state['cases'].append(row);save(a.output/'state.json',state)
        try:
          candidate=load(source,'roof_search_'+config['id'].replace('-','_'))
          for size in plan['shapes']:
           for dt in plan['dtypes']:
            generator=torch.Generator().manual_seed(20261027);dtype=getattr(torch,dt)
            def random(shape):return torch.randn(shape,generator=generator).to(dtype=dtype,device='npu')
            x=random((size,size));y=random((size,size));bias=random((size,))
            old,old_result=helper.capture_call(lambda _:parent.run(x,y,bias),None)
            new,new_result=helper.capture_call(lambda _:candidate.run(x,y,bias),None)
            reference=torch.relu(torch.mm(x,y)+bias);torch.npu.synchronize()
            torch.testing.assert_close(new_result,reference,atol=1e-4*size,rtol=1e-3 if dt=='float16' else .016)
            if len(old)!=1 or len(new)!=1:raise ValueError('不是单kernel路径')
            case=directory/(dt+'-'+str(size));case.mkdir();compiler.compiled_evidence(new,case/'compiled')
            times={'parent':[],'candidate':[]};funcs=dict(parent=helper.direct_call(old),candidate=helper.direct_call(new))
            for repeat in range(plan['repetitions']):
                keys=list(funcs) if repeat==0 else list(reversed(funcs))
                value=measure.pair({key:funcs[key] for key in keys},dict(parent=old[0]['jit'].fn.__name__,candidate=new[0]['jit'].fn.__name__),case/('timing-'+str(repeat)))
                for key in funcs:times[key].append(value[key])
            row['measurements'].append(dict(size=size,dtype=dt,times=times,speedup=statistics.median([p/c for p,c in zip(times['parent'],times['candidate'])])))
            save(a.output/'state.json',state)
          row.update(status='passed',speedup=statistics.geometric_mean(r['speedup'] for r in row['measurements']))
        except Exception as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error)[-7000:])
            # 编译/数值失败可进入下一项；设备状态异常不继续提交。
            torch.npu.synchronize()
        save(a.output/'state.json',state);print(json.dumps({k:v for k,v in row.items() if k!='measurements'},ensure_ascii=False),flush=True)
    valid=[r for r in state['cases'] if r['status']=='passed'];best=max(valid,key=lambda r:r['speedup']) if valid else None
    state.update(status='completed',best=best['case'] if best else None,best_speedup=best['speedup'] if best else None)
    save(a.output/'state.json',state)

if __name__=='__main__':main()
