"""Check optimized waiting-time subtraction against an independent endpoint model."""
import random
import unittest
from hgkda.regions import subtract, union


class IntervalDifference(unittest.TestCase):
    def test_covering_multiple_waits(self):
        self.assertEqual(subtract([(0,2),(4,6),(8,10)],[(1,9)]),[[0.,1.],[9.,10.]])

    def test_random_overlaps_against_endpoint_partition(self):
        rng=random.Random(741)
        for _ in range(200):
            waits=[sorted(rng.sample(range(40),2)) for _ in range(rng.randrange(12))]
            busy=[sorted(rng.sample(range(40),2)) for _ in range(rng.randrange(12))]
            edges=sorted({x for interval in waits+busy for x in interval})
            expected=[]
            for a,b in zip(edges,edges[1:]):
                mid=(a+b)/2
                if any(l<mid<h for l,h in waits) and not any(l<mid<h for l,h in busy):expected.append([a,b])
            self.assertEqual(subtract(waits,busy),union(expected))

    def test_long_disjoint_events(self):
        n=20000
        self.assertEqual(len(subtract([(4*i,4*i+3) for i in range(n)],[(4*i+1,4*i+2) for i in range(n)])),2*n)


if __name__=='__main__':unittest.main()
