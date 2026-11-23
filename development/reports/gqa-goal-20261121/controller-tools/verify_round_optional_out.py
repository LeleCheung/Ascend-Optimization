"""Check archived Python control flow for the documented optional out argument."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path.cwd()
    source=next((root/'development/experiments/lineage/20261117/links-v9/artifacts').glob('e67b73611626*.py'))
    tree=ast.parse(source.read_bytes())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('run','_slow')]
    assert len(functions)==2
    # Compile only the two exact host functions; no imports or device kernel execute.
    scope={'_FAST':[None,None,-1,-1,0,0,1,None,None,None,None]}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    class Input:
        dtype='synthetic_dtype'
        def numel(self):return 1
        def is_contiguous(self):return True
    error=None
    try:
        scope['run'](Input())
    except AttributeError as exc:
        error=str(exc)
    assert error and 'NoneType' in error and 'is_contiguous' in error
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                status='failed',passed=0,total=1,case='nonempty_input_out_omitted',error=error,
                scope='Exact archived run/_slow AST, metadata-only synthetic input; fails before any device operation. Public out=None default must allocate independent output.')
    path=root/'development/experiments/lineage/20261117/current-checks/old-round-optional-out.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
