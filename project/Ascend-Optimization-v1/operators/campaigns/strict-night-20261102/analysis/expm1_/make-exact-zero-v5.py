"""避免8192块额外零选择的UB占用；以不融合乘加保留原小值公式的零符号。"""
import ast
import hashlib
import json
from pathlib import Path


def main():
    batch=Path(__file__).resolve().parents[2]
    parent=batch/'proposals/expm1_/codex-exact-compile-cache-v4/main.py'
    raw=parent.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='f7ead105fbe03f4ee8f5ffcd42548517c01b11623bd809a1b2f5f4b06a2327db'
    text=raw.decode('utf8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_expm1_value_narrow')
    fragment=ast.get_source_segment(text,node)
    assert fragment.count('return tl.where(x == 0.0, x, value)')==1
    assert fragment.count('return tl.where(xf == 0.0, xf, value)')==1
    corrected=fragment.replace('return tl.where(x == 0.0, x, value)','return value').replace(
        'return tl.where(xf == 0.0, xf, value)','return value')
    text=text.replace(fragment,corrected)
    old='ck = kernel.warmup(A, n, grid=(1,), **dict(zip(names, cargs)))'
    assert text.count(old)==1
    text=text.replace(old,'''options = dict(zip(names, cargs))
        if kernel is _expm1_exact:
            options['enable_fp_fusion'] = False
        ck = kernel.warmup(A, n, grid=(1,), **options)''')
    text=text.replace('通用路径继承已通过63项补测的公式，整块低精度分支显式保留零符号。',
        '通用路径继承已通过63项补测的公式；整块小值分支保持乘法结构并关闭乘加融合，零符号须经NPU补测确认。')
    ast.parse(text)
    folder=batch/'proposals/expm1_/codex-exact-zero-v5';folder.mkdir(parents=True,exist_ok=True)
    output=folder/'main.py';new=text.encode('utf8')
    if output.exists():assert output.read_bytes()==new
    else:output.write_bytes(new)
    digest=hashlib.sha256(new).hexdigest()
    spec=dict(operator='expm1_',source_sha256=digest,parent_source_sha256=hashlib.sha256(raw).hexdigest(),
        host_glue=False,exact_expm1_probe=True,attribution='KG/CC提供8192整块与低精度缓冲；Codex修复编译返回类型及失败缓存，依据192KiB UB溢出证据移除整块额外零选择并关闭该核的乘加融合，保留通用路径原公式、对齐保护与严格Triton回退。')
    path=folder/'spec.json';data=(json.dumps(spec,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    # 不修改已执行检查器，派生一个明确设置融合选项的新检查器。
    checker=Path(__file__).with_name('compile-exact-v1.py').read_text(encoding='utf8')
    assert checker.count('options=dict(num_warps=32)')==1
    checker=checker.replace('options=dict(num_warps=32)','options=dict(num_warps=32,enable_fp_fusion=False)')
    checker=checker.replace('核对毫秒级退化是否来自被吞掉的编译失败','核验v5关闭融合后的8192整块核')
    path=Path(__file__).with_name('compile-exact-v5.py');data=checker.encode('utf8')
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    print(json.dumps(spec,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
