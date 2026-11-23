"""Synchronize the original local workflow attachment with the meeting diagram."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sync(root, target):
    meeting = root / 'development/reports/group-meeting-20261119'
    svg_path = meeting / 'research-routes.svg'
    svg = svg_path.read_text(encoding='utf8')
    assert '继续模拟器重放' in svg and '新产物' in svg
    assert '固定源码，修改后编译并继续重放' in svg
    original = target.read_bytes()
    backup = target.with_name(target.stem + '.before-replay-20261120.html')
    if not backup.exists():
        backup.write_bytes(original)
    page = target.read_text(encoding='utf8')
    if '<div class="mermaid">' in page:
        page, count = re.subn(r'  <div class="mermaid">.*?  </div>\s*<script type="module">.*?</script>',
            '  <div class="workflow">' + svg + '</div>', page, flags=re.S)
        assert count == 1
        page = page.replace('.mermaid { background: #fff; padding: 20px; border-radius: 12px; }',
            '.workflow { background: #fff7e8; padding: 16px; border-radius: 12px; }\n'
            '    .workflow svg { display: block; width: 100%; height: auto; }')
        page = page.replace('background: #f7f9fc;', 'background: #fff7e8;')
    else:
        page, count = re.subn(r'<div class="workflow">.*?</div>',
            '<div class="workflow">' + svg + '</div>', page, flags=re.S)
        assert count == 1
    updated = page.encode('utf8')
    target.write_bytes(updated)
    receipt = dict(timestamp=datetime.now(timezone.utc).isoformat(), target=str(target),
        source_svg=str(svg_path), source_svg_sha256=sha(svg_path.read_bytes()),
        backup=str(backup), backup_sha256=sha(backup.read_bytes()),
        previous_target_sha256=sha(original), updated_target_sha256=sha(updated),
        offline_inline_svg=True,
        scope='Original workflow attachment synchronized with the reviewed meeting replay loop; historical attachment preserved.')
    (meeting / 'original-workflow-sync-20261120.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    receipt = sync(args.root.resolve(), args.target.resolve())
    print(json.dumps(dict(status='workflow_synchronized', target=receipt['target'],
        sha256=receipt['updated_target_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
