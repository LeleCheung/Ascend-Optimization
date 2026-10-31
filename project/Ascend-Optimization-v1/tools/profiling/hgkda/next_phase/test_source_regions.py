"""热点聚合使用来源核验的源码范围，避免只看一条指令。"""
import hashlib
import unittest

from regions import analyze


SOURCE='''def kernel():
    for i in range(4):
        x = load(i)
        if x:
            x = x + 1
        store(i, x)
    result = 0
    return result
'''


def listing():
    return dict(kernels=[dict(name='k',cores=[dict(name='c',instructions=[
        dict(pc=hex(16+index*4),instruction=op,running_time_us=1,
             source=dict(file='/compile/main.py',line=line))
        for index,(line,op) in enumerate([(3,'LOAD'),(5,'ADD'),(6,'STORE'),(8,'RET')])])])])


class SourceRegionTests(unittest.TestCase):
    def test_load_compute_store_in_same_loop_region(self):
        sources={'/compile/main.py':dict(text=SOURCE,sha256=hashlib.sha256(SOURCE.encode()).hexdigest())}
        result=analyze(listing(),sources=sources)
        loop=next(r for r in result['regions'] if r['kind']=='source_loop')
        self.assertEqual((loop['start'],loop['end'],loop['instructions']),(2,6,3))
        self.assertEqual(loop['source_lines'],[3,5,6])
        self.assertTrue(any(r['kind']=='source_function' for r in result['regions']))

    def test_wrong_identity_rejected_and_different_path_not_guessed(self):
        with self.assertRaises(ValueError):analyze(listing(),sources={'/compile/main.py':dict(text=SOURCE,sha256='wrong')})
        sources={'/other/main.py':dict(text=SOURCE,sha256=hashlib.sha256(SOURCE.encode()).hexdigest())}
        result=analyze(listing(),sources=sources)
        self.assertTrue(all(r['kind']=='source_neighborhood' for r in result['regions']))

    def test_pc_padding_does_not_break_trace_mapping(self):
        trace=dict(traceEvents=[dict(ph='M',name='process_name',pid=1,args=dict(name='VECTOR')),
                               dict(ph='X',name='ADD',pid=1,ts=0,dur=2,args=dict(pc_addr='0X00000010'))])
        result=analyze(listing(),trace,kernel='k',core='c')
        self.assertEqual(result['regions'][0]['trace_interval_union'],2)


if __name__=='__main__':unittest.main()
