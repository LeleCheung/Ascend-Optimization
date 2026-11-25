"""Build content-addressed KGS contract bundles with canonical tar metadata under operator-bundle/v1."""
import hashlib
import io
from pathlib import Path
import tarfile


def pack(directory, *, exclude=()):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('Contract directory does not exist')
    paths = []
    for path in sorted(root.rglob('*')):
        if path.relative_to(root).as_posix() in exclude:
            continue
        if path.is_symlink():
            raise ValueError('Contract bundles cannot contain symbolic links')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
            paths.append(path)
    if not paths:
        raise ValueError('Contract bundle is empty')
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
