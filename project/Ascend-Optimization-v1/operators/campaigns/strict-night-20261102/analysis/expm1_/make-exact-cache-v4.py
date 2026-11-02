"""按已定位的编译返回类型错误修正整块提案，保留严格Triton与原冻结合同。"""
import ast
import hashlib
import json
from pathlib import Path

BATCH=Path(__file__).resolve().parents[2]
PARENT='cd6d0f1ad8ed5ec79b362e619981b92cfd365c9675b0d24b9bb7dc4afe85d03d'
MATH_PARENT='7696add46b061fd8b21ba055df26967976cf7a91444556217bbcc42a4b31fa81'


def function(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)


def replace_function(text,name,replacement):
    node=function(text,name)
    lines=text.splitlines(True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])


def main():
    source=BATCH/'admissions/expm1_'/PARENT/'main.py'
    original=source.read_bytes()
    assert hashlib.sha256(original).hexdigest()==PARENT
    text=original.decode('utf8')
    doc=ast.parse(text).body[0]
    assert isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant)
    lines=text.splitlines(True)
    text='''"""严格Triton原地expm1：修复KG第二轮整块核编译失败与重复尝试。

FP16/BF16与FP32在constexpr分支中分别返回，避免编译器合并不同返回类型。
失败编译只缓存一次；非对齐输入仍走标准Triton入口。
通用路径继承已通过63项补测的公式，整块低精度分支显式保留零符号。
8192分块策略来自KG/CC第二轮；Codex根据正式逐case延迟及离线编译证据修正。
"""
'''+''.join(lines[doc.end_lineno:])
    math_source=BATCH/'admissions/expm1_'/MATH_PARENT/'main.py'
    math_text=math_source.read_text(encoding='utf8')
    assert hashlib.sha256(math_source.read_bytes()).hexdigest()==MATH_PARENT
    text=replace_function(text,'_expm1_value',ast.get_source_segment(math_text,function(math_text,'_expm1_value')))
    text=replace_function(text,'_expm1_value_narrow','''def _expm1_value_narrow(x, NARROW: tl.constexpr):
    xf = x.to(tl.float32)
    e = tl.exp(xf) - 1.0
    if NARROW:
        series = x * (1.0 + 0.5 * x)
        value = tl.where(tl.abs(x) < 0.01, series, e.to(x.dtype))
        return tl.where(x == 0.0, x, value)
    else:
        series = xf * (1.0 + 0.5 * xf)
        value = tl.where(tl.abs(xf) < 0.01, series, e)
        return tl.where(xf == 0.0, xf, value)''')
    build=ast.get_source_segment(text,function(text,'_build_entry'))
    assert build.count('entry = None')==2
    text=replace_function(text,'_build_entry',build.replace('entry = None','entry = False'))
    text=replace_function(text,'run','''def run(A):
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _launch_general(A, n)
        return A
    ptr = A.data_ptr()
    if ptr & 15:
        _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
        return A
    dev = A.device.index
    entry = _ENTRIES.get((A.dtype, n, dev))
    if entry is None:
        entry = _build_entry(A, n, dev)
    if entry is False:
        _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
        return A
    entry[0](entry[3], 1, 1, _current_stream(dev), entry[1], entry[2],
             None, None, None, ptr, n, *entry[4])
    return A''')
    ast.parse(text)
    folder=BATCH/'proposals/expm1_/codex-exact-compile-cache-v4'
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/'main.py'
    raw=text.encode('utf8')
    if output.exists():assert output.read_bytes()==raw
    else:output.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    record=dict(operator='expm1_',source_sha256=digest,parent_source_sha256=PARENT,
        math_parent_source_sha256=MATH_PARENT,host_glue=False,
        attribution='KG/HG-KDA＋CC/DeepSeek提出8192整块/低精度缓冲；Codex用逐case延迟和单CPU编译定位不同返回类型造成的失败回退，修复constexpr分支、失败缓存，并恢复指针对齐与零符号保护。')
    target=folder/'spec.json'
    encoded=(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    if target.exists():assert target.read_bytes()==encoded
    else:target.write_bytes(encoded)
    print(json.dumps(record,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
