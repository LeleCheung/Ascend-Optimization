"""只读旧图接口及设备AiCPU注册线索；符号存在不代表同语义或可用。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(r'fractional|inv_ex|linalg.{0,24}(?:inv|lu)|lu_factor|\bLU\b|\bLu\b|MatrixInverse', re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cann = Path('/usr/local/Ascend/cann-9.0.0')
    roots = [cann/'opp/built-in/op_impl/aicpu', cann/'opp/built-in/op_proto', cann/'opp/built-in/op_proto/inc']
    configs = []
    libraries = []
    seen = set()
    scanned = dict(text_files=0, libraries=0)
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if path.suffix in {'.json', '.ini', '.h', '.proto', '.txt'}:
                scanned['text_files'] += 1
                raw = path.read_bytes()
                matches = [dict(line=i+1, text=line[:2200])
                    for i, line in enumerate(raw.decode('utf8', errors='replace').splitlines())
                    if PATTERN.search(line)]
                if not matches:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                target = args.output/'files'/digest/path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                configs.append(dict(path=str(path), sha256=digest, snapshot=str(target), matches=matches))
            elif '.so' in path.name:
                scanned['libraries'] += 1
                process = subprocess.run(['strings', '-a', str(path)], capture_output=True,
                    timeout=120)
                matches = sorted(set(line[:2400] for line in process.stdout.decode('utf8', errors='replace').splitlines()
                                     if PATTERN.search(line)))
                libraries.append(dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), returncode=process.returncode,
                    matches=matches, stderr=process.stderr.decode(errors='replace')[:1000]))
    result = dict(status='cpu_inventory_complete',
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        roots=[dict(path=str(root), exists=root.exists()) for root in roots], scanned=scanned,
        configs=configs, libraries=libraries, device_executed=False, qualified_CANN=False,
        scope='安装包图定义和设备AiCPU库的注册线索；没有执行算子，不把名称相似、设备AiCPU或宿主CPU回退混为同语义合格参考。')
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], scanned=scanned, matched_configs=len(configs),
        matched_libraries=sum(bool(row['matches']) for row in libraries))))


if __name__ == '__main__':
    main()
