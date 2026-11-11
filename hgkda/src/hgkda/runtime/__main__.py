"""Explicit public runtime actions; orchestration belongs to the calling host."""
import argparse
import subprocess
import sys

ACTIONS={
 'prepare':('prepare',[]),
 'generate':('host_action',['generate']), 'context':('agent_tools',['context']),
 'preflight':('host_action',['preflight']), 'evaluate':('host_action',['evaluate']),
 'pair':('host_action',['pair']), 'simulate':('collect_simulator',[]),
 'compare-ir':('compare_compiled',[]), 'profile-host':('host_profile',[]),
 'probe-timing':('probe_launch',[]), 'probe-cache':('launch_probe',[]),
 'runtime-api':('runtime_api',[]), 'capture-compiled':('capture_compiled',[]),
}
def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=ACTIONS)
 if not args or args[0] in ['-h','--help']:p.print_help();return 0
 action=p.parse_args(args[:1]).action;module,prefix=ACTIONS[action]
 return subprocess.run([sys.executable,'-X','utf8','-m','hgkda.runtime.'+module,*prefix,*args[1:]],check=False).returncode
if __name__=='__main__':raise SystemExit(main())
