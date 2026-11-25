"""Build a standalone, offline simulator evidence viewer from frozen records."""
import json
from pathlib import Path


def build(root):
    report = root / 'development/reports/batchnorm-simulator-20261118'
    data = json.loads((report / 'interactive-evidence.json').read_bytes())
    for item in data:
        import hashlib
        assert hashlib.sha256(item['source'].encode()).hexdigest() == item['source_sha256']
        assert item['timeline']['absolute_time_qualified'] is False
        assert all(0 <= e[1] < e[2] <= 1 for e in item['events'])
    template = Path(__file__).with_name('simulator_demo.html').read_text(encoding='utf8')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    output = report / 'interactive.html'
    output.write_text(template.replace('__EVIDENCE__', payload), encoding='utf8')
    print(output)


if __name__ == '__main__':
    build(Path.cwd())
