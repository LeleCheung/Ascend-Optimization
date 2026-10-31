"""核对多输出分块的输出覆盖、整除条件与源码变换，不运行NPU。"""
import ast,unittest
from templates import CONFIGS,generate,logical

class TemplatesTests(unittest.TestCase):
    def test_source_valid_and_no_placeholder(self):
        for cfg in CONFIGS:
            source=generate('def run(a,b,c): return a\n',cfg);ast.parse(source)
            self.assertNotIn('__BM__',source)
            self.assertIn('k>=2*bk',source)
            self.assertIn('input.is_contiguous()',source)

    def test_tiles_cover_output_once(self):
        for cfg in CONFIGS:
            for m,n in [(512,512),(1024,2048),(768,1536)]:
                bm,bn,tm,tn=[cfg[k] for k in ['bm','bn','tm','tn']]
                if m%(bm*tm) or n%(bn*tn):continue
                nm,nn=m//(bm*tm),n//(bn*tn);seen=[]
                for tile in range(nm*nn):
                    if cfg['group']:
                        width=4*nn;first=tile//width*4;size=min(nm-first,4)
                        pm,pn=first+tile%size,(tile%width)//size
                    else:pm,pn=tile%nm,tile//nm
                    for i in range(tm):
                        for j in range(tn):seen.append((pm*tm+i,pn*tn+j))
                self.assertEqual(len(seen),len(set(seen)))
                self.assertEqual(set(seen),{(i,j) for i in range(m//bm) for j in range(n//bn)})

    def test_dual_m_and_quad_have_expected_reuse(self):
        c={r['id']:r for r in CONFIGS}
        a=logical(c['dual-n-constexpr'],1024,1024,1024)
        b=logical(c['dual-m-reuse-b'],1024,1024,1024)
        self.assertEqual(a['logical_A_elements'],b['logical_B_elements'])
        self.assertEqual(a['logical_B_elements'],b['logical_A_elements'])
        q=logical(c['quad-64x128'],1024,1024,1024)
        self.assertEqual(q['logical_A_elements'],a['logical_A_elements'])
        self.assertEqual(q['logical_B_elements'],a['logical_B_elements'])

    def test_invalid_types_and_budgets_rejected(self):
        for changes in [{'group':False},{'multi':1},{'stages':5},{'tm':4},{'bm':256}]:
            with self.assertRaises(ValueError):generate('def run(a,b,c): return a\n',{**CONFIGS[0],**changes})

if __name__=='__main__':unittest.main()
