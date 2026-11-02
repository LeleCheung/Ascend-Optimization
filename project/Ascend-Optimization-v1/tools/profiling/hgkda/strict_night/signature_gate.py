"""从AST核对人工提案入口签名，提前拒绝参数顺序、种类或默认值错误。"""
import ast
import hashlib
import inspect
import json
from pathlib import Path


def parameters(text):
    bindings = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = bindings.get(node.value.id) if isinstance(node.value, ast.Name) else None
    node = bindings.get('run')
    if not isinstance(node, ast.FunctionDef):
        raise ValueError('run不是可静态核对的显式Python函数或函数别名')
    args = node.args
    positional = args.posonlyargs+args.args
    defaults = [inspect.Parameter.empty]*(len(positional)-len(args.defaults))+list(args.defaults)
    result = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = 'positional_only' if index < len(args.posonlyargs) else 'positional_or_keyword'
        result.append((arg.arg, kind, inspect.Parameter.empty if default is inspect.Parameter.empty else ast.literal_eval(default)))
    if args.vararg:
        result.append((args.vararg.arg, 'var_positional', inspect.Parameter.empty))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        result.append((arg.arg, 'keyword_only', inspect.Parameter.empty if default is None else ast.literal_eval(default)))
    if args.kwarg:
        result.append((args.kwarg.arg, 'var_keyword', inspect.Parameter.empty))
    return result


def validate(text, definition):
    actual = parameters(text)
    expected = definition['parameters']
    if len(actual) != len(expected):
        raise ValueError('参数数目不同：候选%d，合同%d' % (len(actual), len(expected)))
    for (name, kind, default), parameter in zip(actual, expected):
        if (name, kind) != (parameter['name'], parameter['kind']):
            raise ValueError('参数签名不同：合同%s %s，候选%s %s' %
                             (parameter['kind'], parameter['name'], kind, name))
        required = default is inspect.Parameter.empty
        if required != parameter['required']:
            raise ValueError('必需/可选状态不同：'+name)
        if not required and (type(default) is not type(parameter['default']) or default != parameter['default']):
            raise ValueError('默认值不同：'+name)


def admit(source, definition):
    source, definition = Path(source), Path(definition)
    record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  definition_sha256=hashlib.sha256(definition.read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='AST核对显式Python入口；不执行候选，不替代KGS实际ABI绑定或设备评测')
    try:
        validate(source.read_text(encoding='utf8'), json.loads(definition.read_text(encoding='utf8')))
        record['status'] = 'admitted'
    except (ValueError, TypeError, SyntaxError, KeyError) as error:
        record.update(status='rejected', error=type(error).__name__+': '+str(error))
    return record
