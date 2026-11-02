"""复用已编译的大分块自写设备库，验证Hardswish片上复制与分块改进。"""
from pathlib import Path
import shutil
from common import BATCH,read,save,sha


def main():
    previous=BATCH.parent/'remaining-final-20261031'
    build=read(previous/'device-build-event-large-v6.json')
    binary=Path(build['binary'])
    # 构建记录的原始文件一并核验；旧构建不覆盖。
    for key in ['binary_sha256']:
        if key in build:assert sha(binary)==build[key]
    bootstrap=read(BATCH/'bootstrap.json')
    for op in ['hardswish_out','hardswish']:
        if op=='hardswish_out':parent=BATCH/'proposals/hardswish_out/ascendc-out-v2/main.py'
        else:parent=Path(bootstrap[op]['origin'])
        origin=parent.parent;out=BATCH/'proposals'/op/'ascendc-large-v3';out.mkdir(parents=True,exist_ok=True)
        files={}
        for name in ['_hgkda_remaining_host_v4.so','ascendc_host.cpp']:
            p=out/name
            if p.exists():assert sha(p)==sha(origin/name)
            else:shutil.copyfile(origin/name,p)
            files[str(p)]=sha(p)
        p=out/'libhgkda_remaining.so'
        if p.exists():assert sha(p)==sha(binary)
        else:shutil.copyfile(binary,p)
        files[str(p)]=sha(p)
        for name in ['pointwise.cpp','CMakeLists.txt']:
            source=previous/'native-build-event-large-v6'/name;p=out/name
            if p.exists():assert sha(p)==sha(source)
            else:shutil.copyfile(source,p)
            files[str(p)]=sha(p)
        text=parent.read_text(encoding='utf8').replace(str(origin),str(out))
        text=text.replace(sha(origin/'libhgkda_remaining.so'),sha(binary))
        source=out/'main.py'
        if source.exists():assert source.read_text(encoding='utf8')==text
        else:source.write_text(text,encoding='utf8')
        files[str(source)]=sha(source)
        attribution='Codex复用自写Ascend C大分块／缓冲直接写回，原同stream C++桥和历史KG Triton布局回退保留；本轮独立验证'
        save(out/'manifest.json',dict(files=files,source_sha256=sha(source),device_build=build,
             parent=str(parent),parent_sha256=sha(parent),backend='AscendC_with_Triton_layout_fallback',attribution=attribution))
        save(out/'proposal.json',dict(operator=op,id=out.name,source='main.py',source_sha256=sha(source),
             backend='AscendC_with_Triton_layout_fallback',attribution=attribution,
             prediction='采样中旧自写核略慢于CANN；扩大流式分块并复用输入缓冲写回，尝试减少片上复制和循环次数，最终按完整调用与全部补测判断'))
        print('LARGE_PROPOSAL',op,sha(source))


if __name__=='__main__':main()
