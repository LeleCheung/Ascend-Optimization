"""以C++ CANN封装消除诊断用ctypes包装的额外开销，原结果保留。"""
import hashlib
import json
from pathlib import Path


def process(r):
    process_build(r, r.batch/'cann-cpp-build.json')
    revised=r.batch/'cann-cpp-group-norm-v2-build.json'
    if revised.exists():
        process_build(r, revised, operators=['native_group_norm'], revision='v2')
    inline=r.batch/'cann-cpp-group-norm-v3-build.json'
    if inline.exists():
        process_build(r, inline, operators=['native_group_norm'], revision='v3')


def process_build(r,build_file,operators=None,revision='v1'):
    from run import PY,quote,save_json
    if not build_file.exists():
        r.state['cann_cpp_status']='尚无成功构建清单，保留已有参考，不编造三路完成'
        r.save();return
    build=json.loads(build_file.read_text(encoding='utf8'))
    library=Path(build['binary']);raw=library.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=build['binary_sha256']:raise r.module.Attention('CANN C++二进制身份不符')
    artifacts=r.batch/'references/cpp-artifacts';artifacts.mkdir(parents=True,exist_ok=True)
    target=artifacts/library.name
    target.write_bytes(raw)
    cpp=Path(build['source']);(artifacts/cpp.name).write_bytes(cpp.read_bytes())
    save_json(artifacts/'build.json',build)
    loading='''import hashlib,importlib.util,sys
from pathlib import Path
import torch,torch_npu
_BINARY=Path(BINARY_NAME)
if hashlib.sha256(_BINARY.read_bytes()).hexdigest()!=BINARY_SHA:
    raise RuntimeError('CANN C++参考二进制身份不符')
if torch.__version__!=TORCH_VERSION or torch._C._GLIBCXX_USE_CXX11_ABI!=TORCH_ABI:
    raise RuntimeError('CANN C++参考的Torch版本或ABI不符')
_name=MODULE_NAME
_spec=importlib.util.spec_from_file_location(_name,_BINARY)
_cpp=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_cpp)
'''.replace('BINARY_NAME',repr(str(target))).replace('BINARY_SHA',repr(build['binary_sha256'])).replace('TORCH_VERSION',repr(build['torch'])).replace('TORCH_ABI',repr(build['abi'])).replace('MODULE_NAME',repr(build.get('module_name','_hgkda_cann_cpp_v1')))
    bodies={
        'log10_':'def run(A):\n    return _cpp.log10_inplace(A)\n',
        'hardswish_out':'def run(x,out):\n    return _cpp.hardswish_out(x,out)\n',
        'native_group_norm':'def run(input,weight,bias,N,C,HxW,group,eps):\n    return _cpp.group_norm(input,weight,bias,N,C,HxW,group,eps)\n',
    }
    here=Path(__file__).resolve().parent
    for op,body in bodies.items():
        if operators and op not in operators:continue
        out=r.batch/'references'/op/('cpp-'+revision);out.mkdir(parents=True,exist_ok=True)
        source=out/'main.py';content=loading+'\n'+body
        if source.exists() and source.read_text(encoding='utf8')!=content:raise r.module.Attention('CANN C++包装冻结后改变')
        if not source.exists():source.write_text(content,encoding='utf8')
        save_json(out/'provenance.json',dict(reference_kind='CANN_native_aclnn_cpp',build=build,
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,target,artifacts/cpp.name]},
            lifecycle=('先提交同stream宿主队列，query/execute在同一线程；不等待设备，不创建逐调用Python事件'
                       if revision=='v3' else '复用torch_npu OpCommand与aclnn执行器管理；无新增每次同步或Python事件'),
            scope='直接调用CANN计算；使用torch_npu C++元数据与任务队列适配，不调用aten目标算子'))
        def evaluate(op=op,out=out,source=source):
            r.q.run('reference-cpp-'+op,quote([PY,here/'evaluate_reference.py','--operator',op,'--source',source,
                '--leaf',r.batch/r.q.records[op]['catalog_leaf'],'--output',out/'formal','--server',r.q.server,
                '--label','CANN-cpp','--kind','CANN_native_aclnn_cpp']),2100)
            path=out/'formal/CANN-cpp-1.result.json';value=json.loads(path.read_text(encoding='utf8'))
            return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),
                num_workloads=value.get('num_workloads'),result=str(path),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference_kind='CANN_native_aclnn_cpp',provenance=str(out/'provenance.json'),
                native_manifest=str(out/'provenance.json'),scope='同原KGS合同的C++ CANN直接封装；库内源码未核验')
        result=r.job('reference-cpp:'+op+':'+revision,evaluate)
        if result and result.get('status')=='PASSED':
            gates=out/'extra-checks.json'
            qualification=r.job('reference-cpp-extra:'+op+':'+revision,lambda op=op,source=source,gates=gates:r.supplementary(op,source,gates,'reference-cpp-extra-'+op+'-'+revision))
            result.update(extra_checks=str(gates),supplementary_status=qualification)
        if result and result.get('status')=='PASSED' and result.get('supplementary_status')=='passed':
            previous=r.state.get('references',{}).get(op)
            if previous and previous.get('result')!=result['result']:
                history=r.state.setdefault('reference_history',{}).setdefault(op,[])
                if not any(item.get('result')==previous.get('result') for item in history):history.append(previous)
            r.state['references'][op]=result
        r.state['cann_cpp_status']='已构建；三项结果逐项核验，不以构建成功代表评测通过'
        r.save()
