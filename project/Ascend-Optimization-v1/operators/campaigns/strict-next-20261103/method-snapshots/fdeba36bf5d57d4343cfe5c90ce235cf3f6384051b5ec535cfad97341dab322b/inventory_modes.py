"""单CPU冻结本机精度入口和编译选项；不初始化NPU、不修改编译配置。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
OUT = PROJECT / 'operators/campaigns/strict-next-20261103/analysis/native-precision-modes-v1'


def main():
    assert not OUT.exists(), '已有清单，不能覆盖'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    OUT.mkdir(parents=True)
    sdk = Path('/usr/local/Ascend/cann-9.0.0')
    tbe = sdk / 'opp/built-in/op_impl/ai_core/tbe'
    triton = Path(importlib.util.find_spec('triton').origin).parent
    paths = [sdk / 'include/acl/acl_op_compiler.h', sdk / 'include/acl/acl_op.h',
             sdk / 'include/aclnnop/aclnn_expm1.h', triton / 'backends/ascend/compiler.py']
    paths += list((tbe / 'kernel/config/ascend910b').rglob('*expm1*.json'))
    paths += list((tbe / 'impl/ops_legacy/dynamic').glob('expm1*.py'))
    rows = []
    for path in paths:
        if not path.exists():
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        target = OUT / 'files' / digest / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        lines = raw.decode('utf8', errors='replace').splitlines()
        keywords = ['high_precision', 'high_performance', 'impl_mode', 'implMode',
                    'ACL_OP_SELECT_IMPL_MODE', 'ACL_OPTYPELIST_FOR_IMPLMODE',
                    'num_stages:', 'multibuffer:', 'enable_ubuf_saving:', 'enable_auto_bind_sub_block:',
                    'tbe.dsl.vexp', 'def expm1(']
        matches = [dict(line=i + 1, text=line) for i, line in enumerate(lines) if any(k in line for k in keywords)]
        rows.append(dict(path=str(path), sha256=digest, snapshot=str(target), matches=matches))
    value = dict(status='cpu_inventory_complete', files=rows, device_executed=False,
                 options_modified=False, qualified_CANN=False,
                 checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='通用精度选项存在不证明Expm1支持；910B配置及入口源文件与实际运行探测分别核验。')
    (OUT / 'result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status=value['status'], files=len(rows))))


if __name__ == '__main__':
    main()
