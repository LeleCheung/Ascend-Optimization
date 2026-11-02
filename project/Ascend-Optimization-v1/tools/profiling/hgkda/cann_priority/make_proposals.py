"""将已核验Hardswish设备实现扩展为out接口，保留布局回退。"""
import ast
import shutil
from common import BATCH, read, save, sha


def main():
    previous = read(BATCH / 'bootstrap.json')['hardswish']
    parent = BATCH / previous['path']
    text = parent.read_text(encoding='utf8')
    assert sha(parent) == previous['sha256']
    out = BATCH / 'proposals/hardswish_out/ascendc-out-v2'
    out.mkdir(parents=True, exist_ok=True)
    origin = __import__('pathlib').Path(previous['origin']).parent
    manifest = read(origin / 'manifest.json')
    files = {}
    for name in ['_hgkda_remaining_host_v4.so', 'libhgkda_remaining.so', 'pointwise.cpp', 'ascendc_host.cpp', 'CMakeLists.txt']:
        p = origin / name
        if not p.exists():continue
        target = out / name
        if target.exists():assert sha(target) == sha(p)
        else:shutil.copyfile(p, target)
        files[str(target)] = sha(target)
    text = text.replace(str(origin), str(out))
    # 回退直接复用原out源码，因此独立输出stride与部分重叠处理沿用其已验证实现。
    fallback = (BATCH / 'starting-points/hardswish_out.py').read_text(encoding='utf8')
    fallback = fallback.replace('def run(x, out):', 'def _out_fallback(x, out):')
    tree = ast.parse(fallback)
    assert any(isinstance(n, ast.FunctionDef) and n.name == '_out_fallback' for n in tree.body)
    # 两份Triton辅助函数的名称独立，避免覆盖原始回退的全局对象。
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != '_out_fallback'}
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in names:node.id = '_out' + node.id
            return node
        def visit_FunctionDef(self, node):
            if node.name in names:node.name = '_out' + node.name
            return self.generic_visit(node)
    fallback = ast.unparse(Rename().visit(tree))
    text += '\n\n' + fallback + '''

def run(x,out):
    if out.shape!=x.shape:out.resize_(x.shape)
    return _host.execute(x,out,4,_out_fallback)
'''
    ast.parse(text)
    source = out / 'main.py'
    if source.exists():assert source.read_text(encoding='utf8') == text
    else:source.write_text(text, encoding='utf8')
    files[str(source)] = sha(source)
    save(out / 'manifest.json', dict(parent_manifest=manifest, files=files, source_sha256=sha(source),
         backend='AscendC_with_Triton_layout_fallback',
         attribution='Codex复用已核验自写Ascend C Hardswish，改接out；保留已有KG生成Triton布局回退'))
    save(out / 'proposal.json', dict(operator='hardswish_out', id=out.name, source='main.py', source_sha256=sha(source),
         backend='AscendC_with_Triton_layout_fallback',
         parent=str(parent), parent_sha256=sha(parent),
         prediction='对齐连续out直接由同stream C++启动自写设备kernel，减少Python/Triton分派；非连续和偏移继续走原out路径'))
    print('PROPOSED', source)


if __name__ == '__main__':
    main()
