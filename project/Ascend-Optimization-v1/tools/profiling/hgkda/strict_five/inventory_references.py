"""单CPU只读核查原生接口、聚合头文件和算子配置；不初始化设备。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(r'fractional|inv[_]?ex|inverse|linalg.*(?:inv|lu|solve)|\blu[_ ]?(?:factor|solve|unpack)|expm1|uniform', re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '已有只读清点结果不覆盖'
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cann = Path('/usr/local/Ascend/cann-9.0.0')
    roots = [cann/'include/aclnnop', cann/'opp/built-in/op_proto/inc',
             cann/'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b',
             cann/'opp/built-in/op_impl/ai_core/tbe/config/ascend910b']
    headers = []
    configurations = []
    seen = set()
    scanned = dict(headers=0, configurations=0)
    for root in roots:
        if not root.exists():
            continue
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file.suffix not in {'.h', '.json'}:
                continue
            resolved = file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            data = file.read_bytes()
            if file.suffix == '.h':
                scanned['headers'] += 1
                lines = [dict(line=i+1, text=line[:1800])
                         for i, line in enumerate(data.decode('utf8', errors='replace').splitlines())
                         if PATTERN.search(line)]
                if not lines:
                    continue
                row = dict(path=str(file), resolved=str(resolved), sha256=sha(file), matches=lines)
                headers.append(row)
            else:
                scanned['configurations'] += 1
                if not PATTERN.search(file.name):
                    continue
                row = dict(path=str(file), resolved=str(resolved), sha256=sha(file))
                configurations.append(row)
            target = args.output/'files'/row['sha256']/file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row['snapshot'] = str(target)
    symbols = []
    libraries = sorted(set(p.resolve() for p in (cann/'lib64').glob('*opapi*.so*') if p.is_file()))
    for library in libraries:
        process = subprocess.run(['nm', '-D', '--defined-only', str(library)],
                                 capture_output=True, text=True, timeout=120)
        lines = [line for line in process.stdout.splitlines() if PATTERN.search(line)]
        symbols.append(dict(path=str(library), sha256=sha(library), returncode=process.returncode,
            matched_symbols=lines, stderr=process.stderr[:1800]))
    result = dict(status='cpu_inventory_complete', checker_sha256=sha(__file__),
        roots=[dict(path=str(root), exists=root.exists()) for root in roots],
        scanned=scanned, headers=headers, configurations=configurations, symbols=symbols,
        device_executed=False, qualified_CANN=False,
        scope='只证明当前安装包已检查范围内的接口及配置。发现声明或符号不代表910B支持、同语义或数值合格；缺失也不代表所有厂商版本均不支持。')
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], scanned=scanned, matched_headers=len(headers),
        matched_configurations=len(configurations), libraries=len(libraries))))


if __name__ == '__main__':
    main()
