"""Verify archived string-only proposal construction without running its code."""
import ast
import hashlib
import json
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
    archive=Path('/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1')
    script=archive/'tools/profiling/hgkda/threeway/make_proposals.py'
    batch=archive/'operators/campaigns/threeway-20261030'
    tree=ast.parse(script.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    values={};parents={};edges=[];unmatched=[]
    def value(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return value(node.left)+value(node.right)
        if isinstance(node,ast.Subscript):
            if isinstance(node.slice,ast.Slice):
                s=node.slice;key=slice(value(s.lower) if s.lower else None,value(s.upper) if s.upper else None,value(s.step) if s.step else None)
            else:key=value(node.slice)
            return value(node.value)[key]
        if isinstance(node,ast.Call):
            args=[value(a) for a in node.args]
            if isinstance(node.func,ast.Name) and node.func.id=='get':
                op=args[0];path=batch/'starting-points'/(op+'.py');raw=path.read_bytes()
                parents[op]=dict(path=str(path),sha256=digest(raw));return raw.decode('utf8')
            if isinstance(node.func,ast.Attribute) and node.func.attr in ('replace','index'):
                receiver=value(node.func.value)
                if not isinstance(receiver,str):raise ValueError('not a string operation')
                return getattr(receiver,node.func.attr)(*args)
        raise ValueError('unsupported literal operation '+type(node).__name__)
    for node in function.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if isinstance(node.value,ast.Lambda):continue
            values[node.targets[0].id]=value(node.value)
        elif isinstance(node,ast.AugAssign) and isinstance(node.op,ast.Add) and isinstance(node.target,ast.Name):
            values[node.target.id]+=value(node.value)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='proposal':
            op,name,body=[value(n) for n in node.value.args[:3]]
            output=batch/'proposals'/op/name/'main.py'
            if output.read_bytes()!=body.encode('utf8'):
                unmatched.append(dict(operator=op,proposal=name,reason='reconstructed bytes differ; no edge inferred'))
                continue
            edges.append(dict(parent_sha256=parents[op]['sha256'],source_sha256=digest(output.read_bytes()),
                kind='verified_archived_literal_transformation',script=str(script),script_sha256=digest(script.read_bytes()),
                parent=parents[op],output=str(output),call_line=node.lineno))
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':
            continue
        else:raise ValueError('unsupported statement '+type(node).__name__)
    out=Path('/data/hanle/ascend-optimization/runs/lineage-evidence-20261117/literal-proposal-edges.json')
    out.write_text(json.dumps(dict(edges=edges,unmatched=unmatched),indent=2)+'\n');print(json.dumps(dict(edges=len(edges),unmatched=len(unmatched))))


if __name__=='__main__':main()
