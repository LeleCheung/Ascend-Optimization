"""通过显式SSH主机和解释器运行本地Python文件，不保存机器配置。"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path, help='本地Python文件；-表示从标准输入读取')
    parser.add_argument('--host', required=True, help='已配置的SSH主机或别名')
    parser.add_argument('--python', required=True, dest='interpreter', help='远程Python解释器路径')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args(argv)
    if args.host.startswith('-') or args.timeout <= 0:
        parser.error('主机不能以连字符开头，timeout必须为正数')
    marker = '__HGKDA_REMOTE_COMPLETE_'+uuid.uuid4().hex+'__'
    raw = sys.stdin.buffer.read() if str(args.script) == '-' else args.script.read_bytes()
    payload = raw+('\nprint('+repr('\n'+marker)+')\n').encode('utf8')
    command = ['ssh', '-o', 'BatchMode=yes', args.host, shlex.join([args.interpreter, '-'])]
    result = subprocess.run(command, input=payload, capture_output=True, timeout=args.timeout)
    lines = result.stdout.decode('utf8', errors='replace').splitlines()
    for line in lines:
        if line != marker:
            print(line)
    if result.stderr:
        print(result.stderr.decode('utf8', errors='replace'), file=sys.stderr, end='')
    return result.returncode or (0 if marker in lines else 2)


if __name__ == '__main__':
    raise SystemExit(main())
