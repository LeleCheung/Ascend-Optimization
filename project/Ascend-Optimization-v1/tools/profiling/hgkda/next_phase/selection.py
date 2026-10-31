"""搜索轮次按相对 PyTorch 的验收指标排序，不受参考绝对耗时漂移误导。"""
import math


def search_rank(result):
    """返回越小越好的排序键；不完整、被拒绝或无有效成绩的轮次不参与。"""
    if result.get('status') != 'PASSED' or result.get('is_hack'):
        return None
    total = result.get('num_workloads', 0)
    if not total or result.get('num_passed') != total:
        return None
    score = result.get('geo_mean')
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0:
        return None
    timings = [row for row in result.get('per_workload', []) if row.get('phase') == 'timing']
    if not timings or any(row.get('status') != 'PASSED' or
            not isinstance(row.get('latency_ms'), (int, float)) or
            not math.isfinite(row['latency_ms']) or row['latency_ms'] <= 0 for row in timings):
        return None
    latency = math.exp(sum(math.log(row['latency_ms']) for row in timings) / len(timings))
    return -score, latency
