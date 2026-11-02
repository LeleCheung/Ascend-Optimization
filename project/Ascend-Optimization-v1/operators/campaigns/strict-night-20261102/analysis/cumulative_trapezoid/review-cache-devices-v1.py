"""执行原源码中提取的启动缓存类，检查跨设备计划身份；不导入候选或设备库。"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

BATCH = Path(__file__).resolve().parents[2]
SOURCES = [
    '342e0efa224dbe1e9efb2dd4610bd15337e595953d717c0c93f4bf2e776ea44d',
    '4b63b02670af61eb3f612cc0d455fa59a1deeebcd342fb53d3834e7a104e8488',
    '583a7f62453a24e3775a5e0253a5b1936073939c00f08ea3dd73cc7913061e47',
    '044fbb6ce185e4641417eaf732b560f08aaa5058853fc98e8249f129f9caa2e2',
]


class TensorMetadata:
    def __init__(self, device):
        self.device = SimpleNamespace(index=device)
        self.shape = (16, 16)
        self.dtype = 'float16'

    def data_ptr(self):
        return 4096 * (self.device.index + 1)

    def stride(self):
        return (16, 1)


def exercise(class_node):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def compile_and_launch(*args):
                device = args[0].device.index
                calls.append(dict(kind='standard_launch', compiled_device=device, stream_device=device))

                def launch(*params):
                    calls.append(dict(kind='cached_launch', compiled_device=device, stream_device=params[3]))

                return SimpleNamespace(run=SimpleNamespace(metadata=SimpleNamespace(), launch=launch),
                                       function=device, packed_metadata=())
            return compile_and_launch

    context = dict(torch=SimpleNamespace(is_tensor=lambda obj: isinstance(obj, TensorMetadata)),
                   _raw_stream=lambda device: device)
    # 执行实际提取的类体；只替换外部张量元数据与启动器，不改缓存代码。
    tree = ast.parse(ast.unparse(class_node))
    exec(compile(tree, '<candidate-_Cached>', 'exec'), context)
    cached = context['_Cached'](Kernel())
    cached[(1,)](TensorMetadata(0), TensorMetadata(0), 16)
    cached[(1,)](TensorMetadata(1), TensorMetadata(1), 16)
    return dict(calls=calls, entries=len(cached.entries),
                incompatible_plan_reused=any(c['compiled_device'] != c['stream_device'] for c in calls))


def main():
    rows = []
    for digest in SOURCES:
        source = BATCH/'admissions/cumulative_trapezoid'/digest/'main.py'
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        text = raw.decode('utf8')
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == '_Cached')
        original = exercise(node)
        assert original['incompatible_plan_reused']
        fragment = ast.get_source_segment(text, node)
        old = '(a.data_ptr() & 15, a.shape, a.stride(), a.dtype)'
        assert fragment.count(old) == 1
        corrected = ast.parse(fragment.replace(old, '(a.device, a.data_ptr() & 15, a.shape, a.stride(), a.dtype)')).body[0]
        # 用可哈希设备标识核对修正假设，保持原类中的其他控制流。
        original_init = TensorMetadata.__init__
        class Device:
            def __init__(self, index): self.index = index
            def __hash__(self): return hash(self.index)
            def __eq__(self, other): return isinstance(other, Device) and self.index == other.index
        def init(self, device):
            original_init(self, device)
            self.device = Device(device)
        TensorMetadata.__init__ = init
        try:
            repaired = exercise(corrected)
        finally:
            TensorMetadata.__init__ = original_init
        assert not repaired['incompatible_plan_reused'] and repaired['entries'] == 2
        rows.append(dict(source=str(source.relative_to(BATCH)), source_sha256=digest,
                         class_line=node.lineno, original=original, corrected_hypothesis=repaired))
    record = dict(status='confirmed_cache_identity_defect', device_executed=False,
                  candidate_imported=False, checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, scope='CPU执行实际缓存类，证明设备身份冲突；不声称已在第二张NPU复现，也不代表任何数值或性能成绩。')
    output = Path(__file__).with_name('cache-devices-v1.json')
    assert not output.exists(), '审查原件不覆盖'
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=record['status'], sources=len(rows), device_executed=False)))


if __name__ == '__main__':
    main()
