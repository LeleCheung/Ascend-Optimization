"""仅接受绑定完整源码和构建身份、逐函数审查过的Triton宿主启动桥。"""
import ast
import hashlib
import json
from pathlib import Path

from admission import static


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(source, review_path):
    source = Path(source)
    review = json.loads(Path(review_path).read_text(encoding='utf8'))
    assert review['source_sha256'] == sha(source)
    assert review['status'] == 'reviewed_strict_triton'
    assert review['all_branches_reviewed'] and review['dependencies_reviewed']
    bridges = review.get('reviewed_host_bridges')
    assert bridges, '没有人工逐函数审查的宿主桥'
    text = source.read_text(encoding='utf8')
    binaries = {Path(n.value).resolve() for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith('.so')}
    assert binaries == {Path(b['binary']).resolve() for b in bridges}
    files = {}
    for bridge in bridges:
        assert bridge['candidate_sha256'] == sha(source)
        assert bridge['status'] == 'reviewed_host_only' and bridge['all_functions_reviewed']
        assert bridge['device_computation'] == 'Triton_only'
        assert bridge.get('functions'), '没有逐函数审查清单'
        for key, digest_key in [('binary', 'binary_sha256'), ('cpp_source', 'cpp_source_sha256')]:
            path = Path(bridge[key])
            assert sha(path) == bridge[digest_key], '宿主桥源文件或二进制改变'
            files[str(path)] = dict(sha256=sha(path), kind=key)
        if bridge.get('build_record'):
            path = Path(bridge['build_record'])
            assert sha(path) == bridge['build_record_sha256']
            build = json.loads(path.read_text(encoding='utf8'))
            assert build['source_sha256'] == bridge['cpp_source_sha256']
            assert build['binary_sha256'] == bridge['binary_sha256']
            files[str(path)] = dict(sha256=sha(path), kind='build_record')
    # 只豁免这份已签署完整源码中用于校验/导入二进制及缓存模块的标准库依赖。
    # 数值代码、Torch调用和其它外部入口仍执行原检查，不整体跳过AST。
    issues = [issue for issue in static(text)
              if not (issue[1] == '未核验外部依赖'
                      and issue[2] in {'hashlib', 'importlib.util', 'pathlib', 'sys'})]
    assert not issues, issues
    return dict(status='reviewed_host_only', source_sha256=sha(source),
                review_sha256=sha(review_path), files=files,
                scope='人工逐函数审查绑定的元数据和Triton启动桥；不证明任意C++均安全')
