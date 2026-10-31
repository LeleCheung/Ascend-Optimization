"""优化候选与对照实现分开记账；参考分数不能进入最佳候选恢复。"""
PREFIXES=('verify-current:','queued-proposal:','search:','native-candidate-evaluate:')


def optimization_jobs(jobs):
    return {key:value for key,value in jobs.items() if key.startswith(PREFIXES)}
