"""Reconstruct archived scatter repair strings in memory, never execute scripts."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path.cwd();evidence=root/'development/experiments/lineage/20261117'
    control=root.parent/'.control-state/recovery/legacy-repository-control'
    source=next((evidence/'source-evidence/sources').glob('8a67a004fd75*.py'))
    original=source.read_bytes();text=original.decode('utf8');proof=[]
    for name,outvar in [('prepare-scatter-segmented-prototype.py','text'),
                        ('prepare-scatter-manual-probe.py','source'),
                        ('fix-scatter-manual-probe-constexpr.py','s')]:
        path=control/name;tree=ast.parse(path.read_text(encoding='utf-8-sig'));scope={}
        def val(node):
            if isinstance(node,ast.Constant):return node.value
            if isinstance(node,ast.Name):return scope[node.id]
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return val(node.left)+val(node.right)
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Slice):
                s=node.slice;return val(node.value)[slice(val(s.lower) if s.lower else None,val(s.upper) if s.upper else None)]
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                if node.func.attr=='read_text':return text
                if node.func.attr in ('replace','index'):
                    receiver=val(node.func.value)
                    if not isinstance(receiver,str):raise ValueError('not string')
                    return getattr(receiver,node.func.attr)(*[val(a) for a in node.args])
            raise ValueError('unsupported expression')
        for stmt in tree.body:
            if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):
                name_=stmt.targets[0].id
                if name_ in ('text','source','s','helper','needle','old','new','begin','end','replacement'):
                    try:scope[name_]=val(stmt.value)
                    except (ValueError,KeyError):continue
            # Stop at the first actual output statement, before unrelated probe construction.
            if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text'
                   for n in ast.walk(stmt)):
                break
        value=scope[outvar]
        proof.append(dict(script=str(path.relative_to(root.parent)),script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          input_sha256=hashlib.sha256(text.encode()).hexdigest(),output_sha256=hashlib.sha256(value.encode()).hexdigest()))
        text=value
    final=hashlib.sha256(text.encode()).hexdigest()
    # Check against accepted source by digest, not a filename-specific guess.
    manifest=json.loads((root/'operators/scatter_add/manifest.json').read_bytes())
    assert final==manifest['selected']['sha256'],'historical reconstruction differs from accepted source'
    result=dict(edges=[dict(parent_sha256=hashlib.sha256(original).hexdigest(),source_sha256=final,
                           kind='verified_archived_literal_transformation',proof=proof)])
    (evidence/'scatter-repair-edge.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(verified=True,source_sha256=final)))


if __name__=='__main__':main()
