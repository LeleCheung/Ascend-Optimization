import os,sys,pathlib,subprocess
root=pathlib.Path('/data/hanle/ascend-optimization/goal-20261010/FlagGems')
candidate=pathlib.Path('flaggems-master-dim-adapter-v2-20261010.py').resolve()
plugin=candidate.parent/'amin_timer_plugin.py'
plugin.write_text('''import os,traceback

def pytest_collection_modifyitems(session, config, items):
    import benchmark.base as b
    import torch
    import triton.backends.ascend.testing as t
    print("CONFIG",b.Config.mode,flush=True)
    if os.environ.get("AMIN_PROBE_SET_DEVICE") == "1":
        torch.npu.set_device(0)
        print("EXPLICIT_DEVICE_INITIALIZED",flush=True)
    original=b.Benchmark.get_latency
    def latency(self,op,*args,**kwargs):
        print("OP_START",str(op),[(str(x.shape),str(x.dtype)) for x in args if isinstance(x,torch.Tensor)],flush=True)
        try:
            result=original(self,op,*args,**kwargs)
            print("OP_LATENCY",result,flush=True)
            return result
        except BaseException:
            print("OP_FAILED",str(op),flush=True)
            traceback.print_exc();raise
    b.Benchmark.get_latency=latency
''',encoding='utf-8')
env=os.environ.copy();env['GEMS_VENDOR']='ascend'
env['PYTHONPATH']=os.pathsep.join([str(candidate.parent),str(root/'src'),str(root),env.get('PYTHONPATH','')])
bootstrap="import runpy,site,sys;sys.path.extend(p for p in site.getsitepackages() if p not in sys.path);runpy.run_module('pytest',run_name='__main__',alter_sys=True)"
args=['-q','-s','benchmark/test_amin.py','-m','amin','--level','core','--case-id','benchmark/test_amin.py::test_amin[dtype0]::core::float16::0','--record','json','--override','amin:'+str(candidate)+':run']
for stage in ['plain','instrumented','explicit-device']:
    print('STAGE_START',stage,flush=True)
    current=args+['--output',str(candidate.parent/('amin-'+stage+'.json'))]
    if stage!='plain':current+=['-p','amin_timer_plugin']
    env['AMIN_PROBE_SET_DEVICE']='1' if stage=='explicit-device' else '0'
    result=subprocess.run([sys.executable,'-s','-c',bootstrap,*current],cwd=root,env=env,timeout=180)
    print('STAGE_EXIT',stage,result.returncode,flush=True)
