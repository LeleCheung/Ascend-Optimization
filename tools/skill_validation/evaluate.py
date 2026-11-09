"""源码完成外部逐路径审查后，以冻结合同进行一次正式 KGS 验证。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from hgkda.gate import admit


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    config = read(args.config)
    repo = Path(config['repository'])
    parent = Path(config['run_directory'])
    digest = sha(args.source)
    admit(args.source.read_bytes().decode('utf8'), read(args.review))
    preflight = read(parent/'attempts'/digest/'preflight.json')
    assert preflight['status'] == 'passed' and preflight['source_sha256'] == digest
    run = parent/('formal-'+digest[:12])
    run.mkdir(exist_ok=False)
    root = run/'operators/selu_'
    root.mkdir(parents=True)
    source = root/'candidate.py'
    source.write_bytes(args.source.read_bytes())
    shutil.copyfile(args.review, root/'review.json')
    old_job = config['job']
    shutil.copyfile(old_job['archive'], root/'operator.tar')
    request = read(old_job['request'])
    request['implementation']['sources'][0]['content'] = source.read_bytes().decode('utf8')
    request['implementation']['name'] += '_independent_skill_'+digest[:12]
    save(root/'request.json', request)
    # 预检与正式合同共用冻结检查器和case，原件复制保留，不重复跑同一补测。
    supplementary = parent/'attempts'/digest/'supplementary.json'
    value = read(supplementary)
    assert value['source_sha256'] == digest
    assert value['checker_source']['sha256'] == old_job['checks']['supplementary']['checker_sha256']
    shutil.copyfile(supplementary, root/'supplementary.json')
    original = read(Path(old_job['request']).parents[2]/'plan.json')
    job = dict(old_job, source=str(source), source_sha256=digest, review=str(root/'review.json'),
               request=str(root/'request.json'), archive=str(root/'operator.tar'))
    save(run/'plan.json', dict(original, jobs=[job], expected=1,
         policy='独立skill实际生成候选；同源码一次完整正式请求；合同、参考、计时参数保持冻结'))
    command = [sys.executable, str(repo/'tools/regression/run.py'), '--plan', str(run/'plan.json'),
               '--container', config['container'], '--device', str(config['device']),
               '--port', str(args.port), '--lock', config['lock'], '--device-python', config['device_python'],
               '--cann-environment', str(Path(config['sdk'])/'set_env.sh')]
    with (run/'controller.log').open('xb') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=3600)
    for name in ['audit', 'summarize']:
        subprocess.run([sys.executable, str(repo/f'tools/regression/{name}.py'),
                        '--run-directory', str(run), '--output', str(run/(name+'-final.json'))], check=True)
    save(parent/'formal-result-pointer.json', dict(run_directory=str(run), source_sha256=digest,
         result=read(root/'evaluation/result.json')['geo_mean'], audit=read(run/'audit-final.json')))


if __name__ == '__main__':
    main()
