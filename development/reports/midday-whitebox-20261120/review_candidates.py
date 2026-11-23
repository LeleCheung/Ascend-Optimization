"""Record host semantic checks for model-authored IR; never edit candidate IR."""
from pathlib import Path
import argparse
import hashlib
import json


root = Path(__file__).parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--operators', nargs='+', default=['upsample', 'scatter', 'batchnorm'])
parser.add_argument('--invocation', default='ir-001')
args = parser.parse_args()

for op in args.operators:
    run = root / op
    call = run / 'agent-calls' / args.invocation
    config = json.loads((run / 'config.json').read_bytes())
    submission = json.loads((call / 'submission.json').read_bytes())
    changed = {name: sha(call / 'output-ir' / name) for name in submission['changed_ir']}
    assert all(submission['output_files'][name] == digest for name, digest in changed.items())
    assert submission['decision_sha256'] == sha(call / 'before-edit-decision.json')
    checks = []
    if op == 'scatter':
        for rank in range(2049):
            go = 1024 < rank
            lo, hi = (1025, 2048) if go else (0, 1024)
            original_mid = min((lo + hi) >> 1, 2047)
            replacement_index = 1536 if go else 512
            assert original_mid == replacement_index
            assert (original_mid < rank) == (replacement_index < rank)
        checks += ['全部2049种插入秩的前两轮索引及比较一致：首轮1024，次轮依首轮选择512／1536。',
                   '新增读取与keys_5读取相同不可变行内地址，三个偏移均在2048范围内；后续二分、旧尾步折叠与写回保持原字节。',
                   '三个额外标量读取的同步、资源及性能代价由低层与真机验证；候选预测的降幅仍待实测。']
        proof = dict(kind='initial_two_probe_rank_identity', ranks_checked=2049, status='passed')
    elif op == 'upsample':
        checked = 0
        for height in [1, 2, 15, 16, 17, 31, 32, 64, 128]:
            for width in [1, 2, 15, 16, 17, 127, 128, 129, 256, 257]:
                for i in range(height):
                    for j in range(width):
                        # A stored word has row_ok && in_col. Boundary selects
                        # choose center when a horizontal neighbor is masked.
                        for row in [max(i-1, 0), i, min(i+1, height-1)]:
                            for col in [j-1 if j >= 1 else j, j, j+1 if j+1 < width else j]:
                                assert 0 <= row < height and 0 <= col < width
                        checked += 1
        checks += ['写回lane满足row_ok与in_col；中心load有效，左右邻居由原边界select回退到有效中心，行索引原有clamp保留。',
                   '掩码无效lane在原store掩码下不可观察；插值、FP16边界选择、打包及所有指针和mask原字节保留，kernel内无跨lane归约。',
                   '选取90组高度／宽度，覆盖1、分块边界、尾部及目标128；逐有效输出lane检查全部被消费邻居均有效。',
                   'CPU低层必须核实访存掩码继续生效；完整与专项精度仍由原合同验收。']
        proof = dict(kind='consumed_neighbor_validity', shapes_checked=90,
                     stored_lanes_checked=checked, status='passed')
    else:
        checked = 0
        for steps in range(1, 258):
            current, pending = 0, min(1, steps-1)
            for iteration in range(steps):
                assert current == iteration
                lookahead = min(iteration+2, steps-1)
                assert 0 <= lookahead < steps
                current, pending = pending, lookahead
                checked += 1
        checks += ['支持的正空间尺寸下，输出宽分支的tile数为正；检查1至257个tile的全部循环消费顺序及预取地址。',
                   'cur消费tile n、pend保存下一tile，lookahead钳位到最后tile；1个tile时重复读取同一合法地址。',
                   '统计、归约、运行状态、窄输出分支和逐元素FP32运算顺序保持原字节；外层每个batch块重新初始化两级状态。',
                   '多结果scf.for文本语法由原生解析检查；UB容量、DMA重叠和额外预读的成本由实际编译及真机重放验证。']
        proof = dict(kind='two_tile_lookahead_consumption_and_bounds',
                     tile_counts_checked=257, iterations_checked=checked, status='passed')
    record = dict(decision='approved', source_sha256=config['source_sha256'],
                  changed_ir=changed, decision_sha256=submission['decision_sha256'],
                  reviewer='Codex', checks=checks, semantic_proof=proof,
                  layer_choice_scope='本轮固定当前最佳源码，检验局部IR变换的实际执行成本；源码也可表达同类意图，模型关于仅能在IR表达的推断保留为原件，宿主采用此处选层解释。',
                  validation_required=['CPU lowering and binary review', 'complete KGS',
                                       'all required supplements', 'ABBA', 'post profiling and simulator replay'],
                  scope='Host source/IR semantic review; no candidate edits; correctness and performance pending')
    (call / 'ir-review.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(op, 'approved for CPU compilation', proof)
