"""Prepare an independent regression from accepted snapshots without restarting historical controllers or changing source/contracts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

from hgkda.adapters.bundle import pack
from hgkda.gate import admit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--archive-project', type=Path, required=True)
    parser.add_argument('--original-project', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve()
    old = args.archive_project.resolve()
    original = args.original_project
    output = args.run_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads((repo / 'development/experiments/index.json').read_bytes())
    links, jobs, errors = {}, [], []

    def archived(path):
        return old / Path(path).relative_to(original)

    def dependencies(path):
        """Enumerate explicit absolute-path dependencies, recording only existing archived files."""
        for node in ast.walk(ast.parse(path.read_bytes())):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if not value.startswith(str(original) + '/'):
                continue
            target = archived(value)
            if target.is_file():
                record = dict(path=value, archived=str(target), sha256=sha(target))
                if value in links:
                    assert links[value] == record
                links[value] = record

    for row in index['rows']:
        if not row['accepted']:
            continue
        op = row['operator']
        try:
            root = output / 'operators' / op
            root.mkdir(parents=True)
            source = repo / row['source']['path']
            assert sha(source) == row['source']['sha256']
            review = json.loads((repo / row['review']['path']).read_bytes())
            admit(source.read_bytes().decode('utf8'), review)
            candidate = root / 'candidate.py'
            candidate.write_bytes(source.read_bytes())
            shutil.copyfile(repo / row['review']['path'], root / 'review.json')
            dependencies(candidate)
            checks = {}
            for kind in ['supplementary', 'cann_supplementary']:
                prior = json.loads((repo / row['evidence'][kind]['artifact']['path']).read_bytes())
                assert prior['source_sha256'] == row['source']['sha256']
                command = prior['reproduce_command']
                leaf = archived(command[command.index('--leaf') + 1])
                checker = archived(prior['checker_source']['path'])
                assert sha(checker) == prior['checker_source']['sha256']
                local_checker = root / (kind + '-checker.py')
                local_checker.write_bytes(checker.read_bytes())
                local_leaf = root / (kind + '-catalog')
                shutil.copytree(leaf, local_leaf, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                dependencies(local_leaf / 'oracle.py')
                checks[kind] = dict(checker=str(local_checker), checker_sha256=sha(local_checker),
                                    leaf=str(local_leaf), previous_passed=prior['passed'],
                                    previous_total=prior['total'],
                                    contract_sha256={name:sha(local_leaf/name) for name in
                                                     ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']})
            old_path = Path(row['evidence']['cann']['artifact']['original_path'])
            old_result = archived(old_path if old_path.is_absolute() else repo / old_path)
        except Exception as error:
            errors.append(dict(operator=op, error=type(error).__name__+': '+str(error)))
            continue
        try:
            request_name = ('request.json' if old_result.name == 'result.json'
                            else old_result.name.replace('.result.json', '.request.json'))
            old_request = old_result.with_name(request_name)
            reconstructed = not old_request.exists()
            if reconstructed:
                # Historical KG snapshots retain solutions, contract identities, and timing settings without a separate HTTP request file.
                identity = json.loads((old_result.parent/'identity.json').read_bytes())
                contract = json.loads((old_result.parent/'evaluation-contract.json').read_bytes())
                solution = json.loads((old_result.parent/'solution.json').read_bytes())
                request = dict(api_version='v6.2',
                    binding=dict(bundle_id=contract['bundle_id'],definition=op),
                    implementation={**solution,'language':'python'},
                    settings=identity['retest_contract']['settings'])
                old_request = old_result.parent/'identity.json'
            else:
                request = json.loads(old_request.read_bytes())
            old_content = request['implementation']['sources'][0]['content']
            content = candidate.read_bytes().decode('utf8')
            assert old_content.replace('\r\n','\n') == content.replace('\r\n','\n')
            raw, digest = pack(Path(checks['cann_supplementary']['leaf']))
            assert request['binding']['bundle_id'] == 'sha256:' + digest, 'Contract archive differs from historical formal request'
            (root/'operator.tar').write_bytes(raw)
            request['implementation']['sources'][0]['content'] = content
            save(root/'request.json', request)
            job = dict(operator=op, group=row['group'], order=row['order'], source=str(candidate),
                       source_sha256=sha(candidate), review=str(root/'review.json'),
                       request=str(root/'request.json'), archive=str(root/'operator.tar'),
                       bundle_id='sha256:'+digest, checks=checks,
                       previous_cann_score=row['evidence']['cann']['score'],
                       previous_pytorch_score=row['evidence']['pytorch']['score'],
                       reference_kind=row['reference_kind'], previous_request=str(old_request),
                       request_reconstructed_from_immutable_snapshot=reconstructed,
                       source_line_endings_preserved=old_content != content,
                       settings=request['settings'])
            jobs.append(job)
        except Exception as error:
            errors.append(dict(operator=op, error=type(error).__name__+': '+str(error)))
    save(output/'plan.json', dict(schema_version='hgkda.regression/v1', repository=str(repo),
         index_sha256=sha(repo/'development/experiments/index.json'), expected=index['accepted'], jobs=jobs,
         errors=errors, compatibility_files=list(links.values()),
         policy='Fixed candidates/contracts; one fresh full KGS call per operator and independent original supplementary checks; preserve historical results'))
    print(json.dumps(dict(prepared=len(jobs), expected=index['accepted'], errors=errors,
                         compatibility_files=list(links.values())), ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
