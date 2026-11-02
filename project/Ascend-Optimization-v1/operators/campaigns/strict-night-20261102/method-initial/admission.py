"""新KG覆盖层的候选准入；静态扫描和运行时检查不冒充人工证明。"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


IMPORTS = {'torch', 'torch_npu', 'triton', 'math', 'functools', 'typing',
           'collections', 'itertools', 'operator', 'dataclasses', 'enum'}
METHODS = {'copy_', 'fill_', 'zero_', 'clone', 'contiguous', 'reshape', 'flatten',
           'to', 'cpu', 'cuda', 'numpy', 'resolve_conj', 'resolve_neg'}
TORCH = {'empty', 'empty_like', 'empty_strided', 'Tensor', 'dtype', 'device',
         'finfo', 'iinfo', 'is_tensor', 'get_default_dtype', 'broadcast_shapes'}
FORBIDDEN = {'eval', 'exec', '__import__', 'compile', 'open', 'getattr_dynamic'}


def static(text):
    tree = ast.parse(text)
    errors = []
    aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name] = a.name
                if a.name.split('.')[0] not in IMPORTS:
                    errors.append((n.lineno, '未核验外部依赖', a.name))
        if isinstance(n, ast.ImportFrom):
            m = n.module or ''
            if n.level or m.split('.')[0] not in IMPORTS:
                errors.append((n.lineno, '未核验外部依赖', m))
            for a in n.names:
                aliases[a.asname or a.name] = m + '.' + a.name
    def name(n):
        if isinstance(n, ast.Name):
            return aliases.get(n.id, n.id)
        if isinstance(n, ast.Attribute):
            return name(n.value) + '.' + n.attr
        return ast.unparse(n)
    class Check(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            if any(name(d.func if isinstance(d, ast.Call) else d) == 'triton.jit'
                   for d in n.decorator_list):
                # Triton核内的类型转换、reshape等属于允许的设备计算。
                for x in ast.walk(n):
                    if isinstance(x, ast.Call) and name(x.func).startswith(('torch.', 'torch_npu.')):
                        errors.append((x.lineno, 'Triton核内禁止Torch调用', name(x.func)))
                return
            self.generic_visit(n)
        def visit_Call(self, n):
            call = name(n.func)
            if call in FORBIDDEN:
                errors.append((n.lineno, '禁止动态执行', call))
            if isinstance(n.func, ast.Attribute) and n.func.attr in METHODS:
                errors.append((n.lineno, '宿主Tensor搬运或计算', call))
            if call.startswith('torch.'):
                suffix = call[6:]
                if '.' not in suffix and suffix not in TORCH:
                    errors.append((n.lineno, '未许可Torch计算', call))
                if suffix.startswith(('ops.', 'nn.', 'linalg.', 'special.')):
                    errors.append((n.lineno, '禁止Torch设备计算', call))
                if suffix.startswith('npu.') and suffix.split('.')[1] not in {
                    'current_device', 'current_stream', 'default_generators', 'get_device_properties'}:
                    errors.append((n.lineno, '未许可NPU接口', call))
            if call.startswith('torch_npu.') and call not in {
                'torch_npu._C._npu_getCurrentRawStreamNoWait'}:
                errors.append((n.lineno, '未许可torch_npu设备接口', call))
            if call.endswith(('.load_library', '.load_inline', '.CDLL')):
                errors.append((n.lineno, '未核验二进制入口', call))
            self.generic_visit(n)
    Check().visit(tree)
    return errors


def admit_bundle(bundle, context):
    """在远端preflight/evaluate前执行；缓存只对相同源码、合同与检查器有效。"""
    root = Path(os.environ['HGKDA_STRICT_BATCH'])
    op = context.definition
    text = bundle.kernel_code
    digest = hashlib.sha256(text.encode('utf8')).hexdigest()
    checker = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = Path(os.environ['HGKDA_RUNTIME_AUDIT'])
    leaf = root/'original-catalog/ops'/op
    contract = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in leaf.iterdir() if p.is_file()}
    identity = dict(source_sha256=digest, checker_sha256=checker,
                    runtime_checker_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(), contract=contract)
    folder = root/'admissions'/op/digest
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/'admission.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['identity'] == identity and previous['status'] == 'runtime_passed':
            return previous
    source = folder/'main.py'
    if source.exists() and source.read_text() != text:
        raise RuntimeError('候选SHA缓存不一致')
    source.write_text(text, encoding='utf8')
    issues = static(text)
    record = dict(identity=identity, issues=issues, status='static_rejected' if issues else 'runtime_pending',
                  scope='自动准入仅覆盖已知入口与原输入；最终源码仍需人工逐路径审查')
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    if issues:
        raise ValueError('严格Triton静态准入拒绝：'+str(issues)[:3000])
    output = folder/'runtime-audit.json'
    if output.exists():
        value = json.loads(output.read_text())
    else:
        command = ['/usr/local/python3.11.15/bin/python3.11', str(audit),
                   '--operator', op, '--source', str(source), '--leaf', str(leaf), '--output', str(output)]
        with (folder/'runtime.log').open('ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            # 不超时丢下仍在用卡的子进程；外层watchdog核验整个worker树。
            process.wait()
        if not output.exists():
            raise RuntimeError('严格Triton运行检查无结果；见'+str(folder/'runtime.log'))
        value = json.loads(output.read_text())
    record['runtime'] = str(output)
    record['status'] = 'runtime_passed' if value['status'] == 'passed' else 'runtime_rejected'
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    if record['status'] != 'runtime_passed':
        raise ValueError('严格Triton运行时准入失败：'+str(output))
    return record
