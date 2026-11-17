"""Verify whole-source AST prefixes retained under a renamed fallback entry."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path('development/experiments/lineage/20261117')
    links=json.loads((root/'links-v9/links.json').read_bytes())
    sources={s['sha256']:root/'links-v9'/s['export'] for s in links['sources'] if s['export'].endswith('.py')}
    for p in (root/'source-evidence/sources').glob('*.py'):
        sources.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),p)
    trees={};byast={}
    for h,p in sources.items():
        raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==h
        try:tree=ast.parse(raw)
        except (SyntaxError,UnicodeError):continue
        trees[h]=tree;byast.setdefault(ast.dump(tree),[]).append(h)
    edges=[]
    for child,tree in trees.items():
        for i,node in enumerate(tree.body[:-1]):
            if not isinstance(node,ast.FunctionDef) or node.name not in ('_triton_run','_original_run'):
                continue
            original=node.name;node.name='run'
            key=ast.dump(ast.Module(body=tree.body[:i+1],type_ignores=[]))
            node.name=original
            for parent in byast.get(key,[]):
                if parent!=child:
                    edges.append(dict(parent_sha256=parent,source_sha256=child,
                        kind='whole_source_ast_prefix_with_renamed_entry',entry=original,
                        parent_artifact=str(sources[parent].relative_to(root)),
                        child_artifact=str(sources[child].relative_to(root)),
                        scope='All earlier top-level AST nodes retained in order; only run name changes. Appended wrapper may invoke other backends and is not thereby admitted.'))
    (root/'wrapped-source-edges.json').write_text(json.dumps(dict(edges=edges),indent=2),encoding='utf8')
    print(json.dumps(dict(edges=len(edges))))


if __name__=='__main__':main()
