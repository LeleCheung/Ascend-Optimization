"""执行实际宿主函数的失败路径，核对失败编译是否只尝试一次；无设备库。"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def exercise(source):
    text=source.read_text(encoding='utf8')
    tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'run','_build_entry'}]
    assert len(nodes)==2
    calls=dict(compile=0,fallback=0)
    class Kernel:
        def warmup(self,*args,**kwargs):
            calls['compile']+=1
            raise RuntimeError('fixture: failed compilation')
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):calls['fallback']+=1
            return invoke
    class TensorMetadata:
        dtype='fp16'
        device=SimpleNamespace(index=0)
        def numel(self):return 8192
        def is_contiguous(self):return True
        def data_ptr(self):return 4096
    kernel=Kernel()
    context=dict(_ENTRIES={},_BLOCK=4096,_expm1_flat=kernel,
                 _plan=lambda *args:(kernel,1,(1,8192)))
    for node in nodes:
        exec(compile(ast.parse(ast.get_source_segment(text,node)),'<actual-host-function>','exec'),context)
    tensor=TensorMetadata()
    assert context['run'](tensor) is tensor and context['run'](tensor) is tensor
    return calls


def main():
    batch=Path(__file__).resolve().parents[2]
    paths=[batch/'admissions/expm1_/cd6d0f1ad8ed5ec79b362e619981b92cfd365c9675b0d24b9bb7dc4afe85d03d/main.py',
           batch/'proposals/expm1_/codex-exact-compile-cache-v4/main.py']
    rows=[dict(source=p.relative_to(batch).as_posix(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
               counts=exercise(p)) for p in paths]
    assert rows[0]['counts']==dict(compile=2,fallback=2)
    assert rows[1]['counts']==dict(compile=1,fallback=2)
    result=dict(status='confirmed_failed_compile_cached_once',device_executed=False,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        scope='使用实际run/_build_entry函数与会失败的CPU启动器；核对控制流，不是NPU正确性或性能结果。')
    output=Path(__file__).with_name('failure-cache-v4.json')
    assert not output.exists(), '保留原件'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],counts=[row['counts'] for row in rows])))


if __name__=='__main__':
    main()
