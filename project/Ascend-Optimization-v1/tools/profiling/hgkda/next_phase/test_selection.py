"""验证搜索轮次选择与最终验收使用相同指标。"""
import copy
import unittest

from selection import search_rank


def result(score, latency):
    return dict(status='PASSED', geo_mean=score, num_passed=2, num_workloads=2,
                per_workload=[dict(phase='correctness', status='PASSED'),
                              dict(phase='timing', status='PASSED', latency_ms=latency)])


class SelectionTests(unittest.TestCase):
    def test_relative_improvement_wins_over_lower_absolute_latency(self):
        faster_absolute = result(.75, .095)
        faster_relative = result(.81, .100)
        self.assertIs(min([faster_absolute, faster_relative], key=search_rank), faster_relative)

    def test_tie_uses_absolute_latency(self):
        self.assertLess(search_rank(result(.81, .090)), search_rank(result(.81, .100)))

    def test_incomplete_rejected_or_nonfinite_results_do_not_compete(self):
        valid = result(.81, .100)
        for change in [dict(status='PARTIAL_PASS'), dict(is_hack=True), dict(num_passed=1),
                       dict(geo_mean=None), dict(geo_mean=float('nan')), dict(geo_mean=0)]:
            self.assertIsNone(search_rank(dict(valid, **change)))
        for latency in [None, 0, -1, float('nan'), float('inf')]:
            self.assertIsNone(search_rank(result(.81, latency)))
        missing = copy.deepcopy(valid)
        missing['per_workload'] = []
        self.assertIsNone(search_rank(missing))


if __name__ == '__main__':
    unittest.main()
