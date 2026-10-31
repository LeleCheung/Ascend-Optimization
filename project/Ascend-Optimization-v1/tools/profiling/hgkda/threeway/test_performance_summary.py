import unittest
from performance_summary import aggregate_independent


class PerformanceSummaryTests(unittest.TestCase):
    def test_geometric_average_keeps_below_threshold_measurement(self):
        rows=[dict(status='qualified',source_sha256='a',result='first',score=.64),
              dict(status='qualified',source_sha256='a',result='second',score=1.0)]
        summary=aggregate_independent(rows,'a')
        self.assertAlmostEqual(summary['display_score'],.8)
        self.assertFalse(summary['stable_above_threshold'])

    def test_duplicates_and_other_source_do_not_change_weight(self):
        row=dict(status='qualified',source_sha256='a',result='first',score=.9)
        other=dict(status='qualified',source_sha256='b',result='second',score=9)
        summary=aggregate_independent([row,row,other],'a')
        self.assertEqual(summary['independent_measurements'],1)
        self.assertAlmostEqual(summary['display_score'],.9)


if __name__=='__main__':unittest.main()
