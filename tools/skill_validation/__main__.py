"""服务器集成验证的统一入口；每次只执行显式选择的操作。"""
import argparse
from pathlib import Path
import subprocess
import sys


# 仅分派现有工具；模型调用、设备操作和正式评测仍遵守各自合同。
COMMANDS = {
    'prepare': ('prepare_operator.py', (), '从固定起点准备开发或冻结实验'),
    'generate': ('host_action.py', ('generate',), '调用CC生成下一份候选'),
    'context': ('agent_tools.py', ('context',), '刷新当前源码的模型上下文'),
    'preflight': ('host_action.py', ('preflight',), '审查后执行候选预检'),
    'evaluate': ('host_action.py', ('evaluate',), '审查后执行正式KGS评测并反馈'),
    'pair': ('host_action.py', ('pair',), '采集指定case的候选与参考'),
    'simulate': ('collect_simulator.py', (), '单核模拟器采集、区域时间轴和上下文绑定'),
    'profile-host': ('host_profile.py', (), '按合同采样候选宿主函数栈'),
    'probe-timing': ('probe_launch.py', (), '诊断提交、同步和可选真机任务轨迹'),
    'probe-cache': ('launch_probe.py', (), '诊断冷／热调用缓存与运行钩子'),
    'runtime-api': ('runtime_api.py', (), '提取实际后端启动接口'),
    'compare-ir': ('compare_compiled.py', (), '比较两份已采集编译产物'),
    'check-selu-offset': ('../operators/selu_/check_offset.py', (), 'SELU偏移与存储保护检查'),
    'check-selu-runtime': ('../operators/selu_/check_runtime.py', (), 'SELU当前stream与指针检查'),
    'check-log10-edges': ('../operators/log10_/check_edges.py', (), 'log10特殊值与边界检查'),
    'audit': ('audit_generation.py', (), '离线核验生成轨迹及冻结身份'),
    'finalize': ('finalize_transfer.py', (), '结束实验并保存方法快照与结果'),
    'export': ('export.py', (), '核验凭据后导出实验原件'),
    'verify': ('verify.py', (), '异机离线核验实验原件与正式结果'),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=COMMANDS, help='具体参数使用 <action> --help 查看')
    parser.epilog = '\n'.join(f'{name}: {item[2]}' for name, item in COMMANDS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    action = parser.parse_args(arguments[:1]).action
    script, prefix, _ = COMMANDS[action]
    target = Path(__file__).resolve().parent/script
    return subprocess.run([sys.executable, '-X', 'utf8', str(target), *prefix, *arguments[1:]], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
