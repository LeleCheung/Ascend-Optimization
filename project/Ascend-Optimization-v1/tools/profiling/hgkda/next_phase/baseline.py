"""仅对KGS入口绑定做可核验改名；固定原版计算代码与旧失败原件保留。"""
import ast
import hashlib
import json
from pathlib import Path


class Rename(ast.NodeTransformer):
    def __init__(self,old,new):self.old=old;self.new=new
    def visit_FunctionDef(self,node):
        if node.name==self.old:node.name=self.new
        return self.generic_visit(node)
    def visit_Name(self,node):
        if node.id==self.old:node.id=self.new
        return node


def verify_adapter(original,adapted,spec):
    assert hashlib.sha256(original).hexdigest()==spec['original_sha256'],'原版源码改变'
    assert hashlib.sha256(adapted).hexdigest()==spec['adapted_sha256'],'适配源码改变'
    old,new=spec['renamed_symbol'],spec['private_symbol']
    assert old.isidentifier() and new.isidentifier() and new.startswith('_') and old!=new
    tree=ast.parse(original.decode('utf8'))
    assert new not in {n.id for n in ast.walk(tree) if isinstance(n,ast.Name)},'私有名称冲突'
    expected=Rename(old,new).visit(tree)
    actual=ast.parse(adapted.decode('utf8'))
    assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False),'改动超出入口符号改名'
    return dict(status='verified_symbol_rename_only',original_sha256=spec['original_sha256'],adapted_sha256=spec['adapted_sha256'])


def resolve(batch,operator,record):
    original=batch/record['seed'];raw=original.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['source_sha256'],'冻结原版适配源码改变'
    result=dict(source=original,key='baseline:'+operator,out=batch/'results'/operator/'baseline-fixed',label='FlagGems',source_sha256=record['source_sha256'])
    manifest=batch/'baseline-adapters'/operator/'manifest.json'
    if manifest.exists():
        spec=json.loads(manifest.read_text(encoding='utf8'))
        source=(manifest.parent/spec['source']).resolve();source.relative_to(manifest.parent.resolve())
        verified=verify_adapter(raw,source.read_bytes(),spec)
        suffix=verified['adapted_sha256'][:12]
        result.update(source=source,key='baseline-adapter:'+operator+':'+suffix,
            out=batch/'results'/operator/('baseline-entry-'+suffix),label='FlagGems-entry',
            source_sha256=verified['adapted_sha256'],adapter=dict(manifest=str(manifest),**verified))
    return result
