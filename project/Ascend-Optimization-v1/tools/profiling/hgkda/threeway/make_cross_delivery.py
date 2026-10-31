"""采用CC第五轮设备方案，去掉未校验dim/stride的地址memo并恢复完整别名判断。"""
import hashlib
import json
from make_proposals import BATCH, proposal


def main():
    root=BATCH.parents[5]
    rounds=root/'runs/threeway-20261030/kg-runs/cross_out-threeway-20261030-attempt-2/stages/optimize/work/.kernelgen/evals'
    origin=rounds/'round-0005/main.py'
    source=origin.read_text(encoding='utf8')
    begin=source.index('    f = _FAST[0]\n',source.index('def run('))
    end=source.index('\n    pa = input.data_ptr()\n    po = out.data_ptr()\n',begin)
    source=source[:begin]+source[end:]
    source=source.replace('if pa == po or _overlaps(input, out):','if torch._C._is_alias_of(input, out):')
    source=source.replace('if pb == po or _overlaps(other, out):','if torch._C._is_alias_of(other, out):')
    begin=source.index('# Single-entry memo of the last resolved flat dispatch:')
    end=source.index('_RAW_STREAM =',begin)
    source=source[:begin]+source[end:]
    begin=source.index('    # Remember the fully resolved dispatch')
    end=source.index('\n\ndef _overlaps',begin)
    source=source[:begin]+source[end:]
    assert '_FAST' not in source
    assert 'lo2 = live & (base >= 2) & (base < n_elem)' in source
    assert 'lo1 = live & (base >= 1) & (base < n_elem)' in source
    name='validated-shifted-loads-v5'
    proposal('cross_out',name,source,
        'CC/DeepSeek移位连续读取、去整数取模和4096大块方案，加Codex动态dim/布局及别名保护修复',
        '保留设备计算优化，每次读取真实元数据和当前stream；不按地址跳过dim/stride检查，所有读取有真实范围掩码')
    manifest=BATCH/'proposals/cross_out'/name/'proposal.json'
    value=json.loads(manifest.read_text(encoding='utf8'))
    value.update(origin=dict(source=str(origin),source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest()),
        attribution='CC/DeepSeek生成去取模/去gather、分块及读边界修复；Codex提供诊断/失败记录、移除错误地址memo并恢复storage别名保护；原KGS及追加检查验收')
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    rejected={}
    for number in [2,3,4,5]:
        path=rounds/('round-%04d'%number)/'main.py'
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        rejected[sha]=dict(source=str(path),status='rejected_by_source_review',
            reason=('后向移位读取缺少上界，尾块存在越界读取' if number in [2,3] else
                    '地址memo未复核dim/stride，固定地址改变dim仍会错误复用旧分派；逻辑numel字节范围不足以保护strided共享storage'))
    (BATCH/'disqualified-sources.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    old=BATCH/'proposals/cross_out/safe-shifted-loads-v4/proposal.json'
    if old.exists():old.rename(old.with_name('proposal.withdrawn.json'))
    print(manifest)


if __name__=='__main__':main()
