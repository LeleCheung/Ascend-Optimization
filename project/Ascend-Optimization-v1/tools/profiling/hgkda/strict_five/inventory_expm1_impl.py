"""只读核对安装包是否保留可选择精度的Expm1实现；不设置编译或运行选项。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    base = Path('/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe')
    roots = [base/'impl', base/'ascendc']
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*expm1*')):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            row = dict(path=str(path), resolved=str(path.resolve()), bytes=len(raw), sha256=digest)
            if path.suffix in {'.py', '.h', '.hpp', '.cpp', '.json'}:
                target = args.output/'files'/digest/path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                row['snapshot'] = str(target)
                row['precision_mentions'] = [dict(line=i+1, text=line[:1600])
                    for i, line in enumerate(raw.decode('utf8', errors='replace').splitlines())
                    if any(word in line.lower() for word in ['precision', 'expm1', 'taylor', 'exp(', 'mode'])]
            files.append(row)
    result = dict(status='cpu_inventory_complete',
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        roots=[dict(path=str(root), exists=root.exists()) for root in roots], files=files,
        device_executed=False, options_modified=False, qualified_CANN=False,
        scope='只检查安装包TBE及Ascend C目录内名称匹配的实现文件；不改变精度、确定性、环境或共享服务。源码精度分支存在也不代表真机参考合格。')
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], files=len(files))))


if __name__ == '__main__':
    main()
