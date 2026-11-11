"""内部方法实验的核验与归档入口；设备和模型动作见hgkda.runtime。"""
import argparse,subprocess,sys
from pathlib import Path
COMMANDS={"prepare-history":"prepare_operator.py","audit":"audit_generation.py","finalize":"finalize_transfer.py","export":"export.py","verify":"verify.py"}
def main():
 args=sys.argv[1:];p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=COMMANDS)
 if not args or args[0] in ["-h","--help"]:p.print_help();return 0
 action=p.parse_args(args[:1]).action
 return subprocess.call([sys.executable,"-X","utf8",str(Path(__file__).parent/COMMANDS[action]),*args[1:]])
if __name__=="__main__":raise SystemExit(main())
