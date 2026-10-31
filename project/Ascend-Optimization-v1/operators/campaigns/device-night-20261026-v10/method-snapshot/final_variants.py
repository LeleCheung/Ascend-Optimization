"""正式候选共享的流水适用条件；开发探针模板和失败原件保持不变。"""
from variants import transform as probe_transform


def transform(source,op,config):
    text=probe_transform(source,op,config)
    if op=='matmul_bias_activation':
        old='if not (m>0 and n>0 and k>0 and m%bm==0'
        assert text.count(old)==1
        text=text.replace(old,'if not (m>0 and n>0 and k>=2*bk and m%bm==0',1)
    return text
