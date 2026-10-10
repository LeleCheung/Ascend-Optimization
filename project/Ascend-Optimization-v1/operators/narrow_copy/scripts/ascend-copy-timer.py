"""仅修复原设备计时器中 N/A kernel_name 的字符串解析，保留全部设备任务。"""
import hashlib
import inspect
import types


def repair_collector(source):
    old = 'df["kernel_name"].str.contains('
    new = 'df["kernel_name"].astype("string").str.contains('
    if source.count(old) != 1:
        raise RuntimeError('安装计时器结构改变，拒绝自动修复')
    return source.replace(old, new)


def make_device_task_timer(testing):
    # 只复制函数 globals；不改 site-packages，不覆盖 testing 的任何全局函数。
    original = inspect.getsource(testing._collect_prof_result)
    namespace = dict(testing.__dict__)
    exec(compile(repair_collector(original), '<ascend-copy-timer>', 'exec'), namespace)
    profiler = testing.do_bench_npu_profiler
    timer = types.FunctionType(profiler.__code__, namespace, profiler.__name__,
                               profiler.__defaults__, profiler.__closure__)
    timer.__kwdefaults__ = profiler.__kwdefaults__
    timer.original_collector_sha256 = hashlib.sha256(original.encode()).hexdigest()
    return timer


_timer = None


def time_device_tasks(fn):
    global _timer
    if _timer is None:
        from triton.backends.ascend import testing
        _timer = make_device_task_timer(testing)
    return _timer(fn)
