"""生成 KGS 内容寻址合同包；按公开 operator-bundle/v1 规则规范化 tar 元数据。"""
import hashlib
import io
from pathlib import Path
import tarfile


def pack(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('合同目录不存在')
    paths = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('合同包不能包含符号链接')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
            paths.append(path)
    if not paths:
        raise ValueError('合同包为空')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            data = path.read_bytes()
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    return raw, hashlib.sha256(raw).hexdigest()
