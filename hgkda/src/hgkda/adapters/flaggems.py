"""Parse native FlagGems test results; process success alone does not establish operator acceptance."""
import math


def summarize(summary, operator, returncode):
    row = summary.get('result', {}).get(operator, {})
    accuracy = row.get('accuracy', {})
    performance = row.get('performance', {})
    points = []
    for dtype, group in performance.get('data', {}).items():
        for shape, value in group.get('details', {}).items():
            points.append(dict(dtype=dtype, shape=shape, **value))
    measured = [p for p in points if all(isinstance(p.get(k), (int, float))
                and math.isfinite(p[k]) and p[k] > 0 for k in ['base', 'gems', 'speedup'])]
    accuracy_passed = (accuracy.get('status') == 'Passed' and accuracy.get('exit_code') == 0
                       and accuracy.get('passed', 0) > 0 and accuracy.get('failed', 0) == 0
                       and accuracy.get('errors', 0) == 0)
    performance_passed = (performance.get('status') == 'Passed' and performance.get('exit_code') == 0
                         and len(measured) > 0 and len(measured) == len(points))
    complete = returncode == 0 and accuracy_passed and performance_passed
    return dict(operator=operator, status='passed' if complete else 'failed',
                accuracy=accuracy, performance=performance,
                measured_points=len(measured), points=points,
                geometric_speedup=(math.exp(sum(math.log(p['speedup']) for p in measured)/len(measured))
                                   if measured else None),
                skipped_accuracy_cases=accuracy.get('skipped', 0),
                scope='upstream_selected_operator_tests_and_core_benchmark',
                pr_ready=False,
                note='Reports native tests for selected operators in the target repository; related regressions, CI, code review, and KGS acceptance remain separate.')
