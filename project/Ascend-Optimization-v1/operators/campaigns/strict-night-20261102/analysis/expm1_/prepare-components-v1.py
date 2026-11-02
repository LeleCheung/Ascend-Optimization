"""冻结已有候选与原PyTorch的小/大输入诊断清单，不直接操作NPU。"""
import hashlib
import json
from pathlib import Path


BATCH = Path(__file__).resolve().parents[2]
PROJECT = BATCH.parents[2]
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote(path):
    return REMOTE + '/' + path.relative_to(PROJECT).as_posix()


def main():
    leaf = BATCH / 'original-catalog/ops/expm1_'
    source = BATCH / 'candidates/expm1_/7696add46b061fd8/main.py'
    assert sha(source) == '7696add46b061fd8b21ba055df26967976cf7a91444556217bbcc42a4b31fa81'
    checker = PROJECT / 'tools/profiling/hgkda/strict_night/component_probe.py'
    cases = [json.loads(line) for line in (leaf / 'timing.jsonl').read_text().splitlines() if line.strip()]
    items = []
    for label, case_name in [('small', 'expm1_-timing-0'), ('large', 'expm1_-timing-3')]:
        case = next(case for case in cases if case['name'] == case_name)
        for side, path in [('candidate', source), ('original_PyTorch', leaf / 'oracle.py')]:
            items.append(dict(name=label + '-' + side, side=side, spec=dict(
                operator='expm1_', source=remote(path), source_sha256=sha(path),
                leaf=remote(leaf), case=case)))
    value = dict(schema='execution-components-queue-v1', operator='expm1_', items=items,
        checker_sha256=sha(checker), oracle_sha256=sha(leaf / 'oracle.py'),
        timing_sha256=sha(leaf / 'timing.jsonl'), qualified_CANN=False,
        formal_performance_measured=False,
        purpose='区分小输入宿主调用与大输入设备执行差距；使用原冻结输入和一次诊断采集，不重测正式成绩。')
    target = BATCH / 'component-diagnostics/expm1-components-v1/spec.json'
    assert not target.exists(), '已有冻结清单，不覆盖'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(status='prepared', items=len(items), checker_sha256=value['checker_sha256'])))


if __name__ == '__main__':
    main()
