"""为已构建Ascend C log10生成带源码/二进制身份的混合候选。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
batch=PROJECT/'operators/campaigns/threeway-20261030'
build=json.loads((batch/'ascendc-build.json').read_text())
assert build['status']=='built'
library=next(Path(p) for p in build['files'] if p.endswith('.so'))
raw=library.read_bytes();digest=hashlib.sha256(raw).hexdigest()
assert digest==build['files'][str(library)]
out=batch/'native-candidates/log10_/ascendc-ln-v1'
out.mkdir(parents=True,exist_ok=True)
(out/'libhgkda_log10.so').write_bytes(raw)
for name in ['log10.cpp','CMakeLists.txt']:
    (out/name).write_bytes((Path(__file__).parent/'ascendc'/name).read_bytes())
body=(batch/'starting-points/log10_.py').read_text(encoding='utf8').replace('def run(A):','def _triton_run(A):',1)
body+='''

# HG-KDA源码自建Ascend C主路径；不调用PyTorch/CANN算子计算log10。
import ctypes as _ctypes
import hashlib as _hashlib
from pathlib import Path as _Path

_NATIVE_LIBRARY = _Path(LIBRARY_PATH)
if _hashlib.sha256(_NATIVE_LIBRARY.read_bytes()).hexdigest() != LIBRARY_SHA:
    raise RuntimeError('Ascend C候选二进制身份不符，请重新构建与冻结')
_NATIVE_HANDLE = _ctypes.CDLL(str(_NATIVE_LIBRARY))
_NATIVE_LAUNCH = _NATIVE_HANDLE.hgkda_log10_launch
_NATIVE_LAUNCH.argtypes = [_ctypes.c_uint32, _ctypes.c_void_p, _ctypes.c_void_p, _ctypes.c_uint32, _ctypes.c_int]
_NATIVE_LAUNCH.restype = None
_NATIVE_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 27}
_FLUSHED_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStream', None)


def run(A):
    n = A.numel()
    if not n:
        return A
    dtype = _NATIVE_DTYPES.get(A.dtype)
    pointer = A.data_ptr()
    if (dtype is None or n >= 2**32 or n % 16 or pointer & 31
            or not A.is_contiguous() or _FLUSHED_STREAM is None):
        return _triton_run(A)
    # 刷新宿主队列使前序PyTorch读写先入同一设备stream，不等待设备完成。
    stream = _FLUSHED_STREAM(A.get_device())
    blocks = min(40, (n + 16383) // 16384)
    _NATIVE_LAUNCH(blocks, stream, pointer, n, dtype)
    return A
'''
body=body.replace('LIBRARY_PATH',repr(str(out/'libhgkda_log10.so'))).replace('LIBRARY_SHA',repr(digest))
source=out/'main.py'
if source.exists() and source.read_text(encoding='utf8')!=body:raise FileExistsError(source)
source.write_text(body,encoding='utf8')
manifest=dict(operator='log10_',backend='source_built_AscendC_with_Triton_layout_fallback',
    implementation_language='python',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,out/'log10.cpp',out/'CMakeLists.txt',out/'libhgkda_log10.so']},
    build=build,scope='主路径包含三种dtype；非连续/非32字节对齐/非16整除输入沿用有效Triton；相同原KGS合同',
    attribution='Codex编写Ascend C候选；Triton布局回退复用历史CC/DeepSeek与Codex实现；尚无新性能结论')
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='prepared',source=str(source),library_sha256=digest)))
