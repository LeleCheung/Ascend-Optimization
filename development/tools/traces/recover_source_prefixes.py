"""Identify complete earlier implementations embedded at the start of later ones.

This proves retained source text, not that every retained function remains active.
No candidate is executed or written. Both original artifact hashes are retained.
"""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path('development/experiments/lineage/20261117')
    links=json.loads((root/'links-v9/links.json').read_bytes())
    sources={s['sha256']:root/'links-v9'/s['export'] for s in links['sources']
             if s['export'].endswith('.py')}
    for path in (root/'source-evidence/sources').glob('*.py'):
        sources.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),path)
    raw={}
    for sha,path in sources.items():
        data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==sha
        raw[sha]=data.rstrip()
    edges=[]
    for parent,data in raw.items():
        if len(data)<500:
            continue
        try:
            tree=ast.parse(data)
        except (SyntaxError,ValueError):
            continue
        has_entry=any((isinstance(n,ast.FunctionDef) and n.name=='run') or
                      (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='run'
                                                       for t in n.targets)) for n in tree.body)
        if not has_entry:
            continue
        for child,later in raw.items():
            if parent==child or len(later)<=len(data) or not later.startswith(data):
                continue
            if later[len(data):len(data)+1] not in (b'\n',b'\r'):
                continue
            edges.append(dict(parent_sha256=parent,source_sha256=child,kind='exact_source_prefix',
                              parent_artifact=str(sources[parent].relative_to(root)),
                              child_artifact=str(sources[child].relative_to(root))))
    (root/'source-prefix-edges.json').write_text(json.dumps(dict(edges=edges),indent=2),encoding='utf8')
    print(json.dumps(dict(edges=len(edges))))


if __name__=='__main__':
    main()
