"""冻结CC生成的移位读取候选，补齐尾块读边界并恢复完整storage别名保护。"""
import hashlib
import json
from make_proposals import BATCH, proposal


def main():
    root=BATCH.parents[5]
    origin=root/'runs/threeway-20261030/kg-runs/cross_out-threeway-20261030-attempt-2/stages/optimize/work/.kernelgen/evals/round-0002/main.py'
    source=origin.read_text(encoding='utf8')
    for old,new in [
        ('lo2 = live & (base >= 2)','lo2 = live & (base >= 2) & (base - 2 < n_elem)'),
        ('lo1 = live & (base >= 1)','lo1 = live & (base >= 1) & (base - 1 < n_elem)'),
        ('if input.data_ptr() == out.data_ptr() or _overlaps(input, out):','if torch._C._is_alias_of(input, out):'),
        ('if other.data_ptr() == out.data_ptr() or _overlaps(other, out):','if torch._C._is_alias_of(other, out):'),
    ]:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    name='safe-shifted-loads-v4'
    proposal('cross_out',name,source,
        'CC/DeepSeek生成的去取模及去gather连续移位读取，加Codex尾块读取上界及storage别名保护修复',
        '保留连续读取性能优势，所有读地址限制于真实输入范围；恢复跨非连续storage视图的别名保护')
    manifest=BATCH/'proposals/cross_out'/name/'proposal.json'
    value=json.loads(manifest.read_text(encoding='utf8'))
    value.update(origin=dict(source=str(origin),source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest()),
        attribution='CC/DeepSeek生成移位读取与精确块内phase方案；Codex提供先前失败证据、修复读取上界与别名保护；原KGS及独立补测验收')
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(manifest)


if __name__=='__main__':main()
