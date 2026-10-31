"""将连续叉积的UB动态gather改为每lane一组3向量的直接寻址。"""
from make_proposals import BATCH, proposal


def main():
    source = (BATCH/'starting-points/cross_out.py').read_text(encoding='utf8')
    begin = source.index('@triton.jit\ndef _cross_flat_kernel')
    end = source.index('\n\n\n# ---', begin)
    kernel = '''@triton.jit
def _cross_flat_kernel(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < n_elem // 3
    off = row * 3
    a0 = tl.load(a_ptr + off, mask=mask, other=0.0)
    a1 = tl.load(a_ptr + off + 1, mask=mask, other=0.0)
    a2 = tl.load(a_ptr + off + 2, mask=mask, other=0.0)
    b0 = tl.load(b_ptr + off, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + off + 1, mask=mask, other=0.0)
    b2 = tl.load(b_ptr + off + 2, mask=mask, other=0.0)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)
    tl.store(o_ptr + off + 1, a2 * b0 - a0 * b2, mask=mask)
    tl.store(o_ptr + off + 2, a0 * b1 - a1 * b0, mask=mask)
'''
    source = source[:begin] + kernel + source[end:]
    source = source.replace('block = _pick_block(n, out.element_size())', 'block = 512')
    source = source.replace('step = (block // 3) * 3', 'step = block * 3')
    proposal('cross_out', 'direct-vector-address-v2', source,
             '连续末维3的叉积改为每lane计算一个向量，直接stride3读写，去掉四次大tile动态gather；别名和通用布局沿用原路径',
             '大输入若受UB gather组织限制应明显降低设备时间；若stride3搬运更差则保留原版。采用固定512向量块，不读取评测身份或输入值')


if __name__ == '__main__':
    main()
