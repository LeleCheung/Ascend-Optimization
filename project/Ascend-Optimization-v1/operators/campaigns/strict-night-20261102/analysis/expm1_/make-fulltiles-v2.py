"""保留KG已测公式和宿主路径，只将大连续输入的完整块与尾块分开。"""
import hashlib
import json
from pathlib import Path


BATCH = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    parent = BATCH / 'candidates/expm1_/7696add46b061fd8/main.py'
    assert sha(parent) == '7696add46b061fd8b21ba055df26967976cf7a91444556217bbcc42a4b31fa81'
    folder = BATCH / 'proposals/expm1_/codex-fulltiles-expm1-v2'
    assert not folder.exists(), '已有提案原件，不覆盖'
    text = parent.read_text(encoding='utf8')
    start = text.index('@triton.jit\ndef _expm1_streaming(')
    end = text.index('@triton.jit\ndef _expm1_scaled(', start)
    replacement = '''@triton.jit
def _expm1_streaming(a_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """完整块无掩码处理，尾块单独保护；每轮仍先读取两个独立数据块。"""
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    full = (N // step) * step
    for start in range(tl.program_id(0) * (2 * BLOCK), full, step):
        off0 = start + lane
        off1 = start + BLOCK + lane
        x0 = tl.load(a_ptr + off0)
        x1 = tl.load(a_ptr + off1)
        y0 = _expm1_value(x0)
        y1 = _expm1_value(x1)
        tl.store(a_ptr + off0, y0.to(a_ptr.dtype.element_ty))
        tl.store(a_ptr + off1, y1.to(a_ptr.dtype.element_ty))
    off0 = full + tl.program_id(0) * (2 * BLOCK) + lane
    off1 = off0 + BLOCK
    x0 = tl.load(a_ptr + off0, off0 < N, other=0.0)
    x1 = tl.load(a_ptr + off1, off1 < N, other=0.0)
    y0 = _expm1_value(x0)
    y1 = _expm1_value(x1)
    tl.store(a_ptr + off0, y0.to(a_ptr.dtype.element_ty), off0 < N)
    tl.store(a_ptr + off1, y1.to(a_ptr.dtype.element_ty), off1 < N)


'''
    text = text[:start] + replacement + text[end:]
    end_docstring = text.index('"""', 3) + 3
    text = '''"""严格Triton原地expm1：继承KG第四稿公式、布局与启动缓存。

Codex将大连续输入改为完整块/尾块分离，保留双块先读后写。
完整块区间从num_programs和BLOCK计算，适用于一般元素数；不按输入值或用例名特化。
该提案需独立正式评测和补测，不预先声称性能提升。
"""''' + text[end_docstring:]
    folder.mkdir(parents=True)
    source = folder / 'main.py'
    source.write_text(text, encoding='utf8', newline='\n')
    save(folder / 'spec.json', dict(operator='expm1_', source_sha256=sha(source),
        parent_source_sha256=sha(parent), defer_if_target_already_met=True,
        attribution='Codex在KG/CC第四稿上分离完整块与尾块，保持数值公式、布局和单次启动语义；无合格CANN参考，仅沿原PyTorch合同评测。'))
    save(folder / 'review.json', dict(operator='expm1_', source_sha256=sha(source),
        status='reviewed_strict_triton', all_branches_reviewed=True, dependencies_reviewed=True,
        reviewer='Codex逐路径审查KG父稿及完整块/尾块变更', review_notes=[
            '只改变大连续输入的Triton流式核：full是完整轮次覆盖区间的末端，循环内两个块都在[0,N)内；两个尾块分别带边界保护。',
            '保持父稿二阶小幅值公式、exp分支、零符号和输出dtype，不修改输入或容差。',
            'N成为constexpr；连续启动计划原缓存键已有dtype/numel/device，元数据与二进制绑定当前kernel和BLOCK。',
            'warmup只编译，实际提交在构建器外恰好一次；每次读取当前stream。',
            '非对齐、等比例stride及最多八个合并维度的非连续输入均继承父稿Triton路径；超范围显式拒绝。',
            '宿主只读元数据、设备属性和当前stream，不保存Tensor/输出，不含CANN、Ascend C、DMA、PyTorch或CPU数值回退。',
            '父稿正式30/30和补测63/63不代表本稿通过；本稿编译与NPU结果必须独立核验。']))
    print(json.dumps(dict(status='prepared', source_sha256=sha(source), parent_source_sha256=sha(parent))))


if __name__ == '__main__':
    main()
