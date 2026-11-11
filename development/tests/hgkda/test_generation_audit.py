"""核对模型工具记录能够解释候选源码变化，防止把外部补丁当作模型生成。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('generation_audit',ROOT/'development/tools/skill_validation/audit_generation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GenerationAuditTests(unittest.TestCase):
    def test_only_frozen_method_tools_are_accepted_as_development_external_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp);tool=package/'tools/skill_validation/tool.py';tool.parent.mkdir(parents=True)
            tool.write_bytes(b'method tool')
            config=dict(phase='development',package=str(package))
            method=dict(files=[dict(path='tools/skill_validation/tool.py',sha256=module.sha(tool))])
            self.assertEqual(module.verify_development_method_read(tool,config,method)['status'],'verified_method_tool')
            self.assertIsNone(module.verify_development_method_read(tool,dict(config,phase='frozen'),method))
            tool.write_bytes(b'changed')
            self.assertIsNone(module.verify_development_method_read(tool,config,method))
            candidate=package/'candidate.py';candidate.write_bytes(b'candidate')
            method['files'].append(dict(path='candidate.py',sha256=module.sha(candidate)))
            self.assertIsNone(module.verify_development_method_read(candidate,config,method))

    def test_existing_candidate_is_explicit_and_wrong_origin_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Path(tmp);(workspace/'starter.py').write_bytes(b'existing generated source')
            digest=module.sha(workspace/'starter.py')
            p=workspace/'starter-provenance.json'
            record=dict(kind='existing_generated_candidate',source_sha256=digest,prior_best_source_provided=True)
            p.write_text(json.dumps(record))
            result=module.starter_identity(workspace)
            self.assertTrue(result['prior_best_source_provided'])
            p.write_text(json.dumps(dict(record,source_sha256='different')))
            with self.assertRaises(AssertionError):module.starter_identity(workspace)

    def test_legacy_shell_replay_is_static_and_rejects_other_code(self):
        command="python - <<'EOF'\nimport re\ns=open('candidate.py').read()\ns=s.replace('old','new')\nopen('candidate.py','w').write(s)\nEOF\ngrep -n old candidate.py"
        self.assertEqual(module.legacy_shell_replacement(command,'old old'),'new new')
        self.assertIsNone(module.legacy_shell_replacement(command.replace('import re','import os; os.system("bad")'),'old'))
        self.assertIsNone(module.legacy_shell_replacement(command.replace("'new'","str(123)"),'old'))

    def check(self,root,output):
        call=root/'call';call.mkdir();workspace=root/'workspace';workspace.mkdir()
        original='value = 1\n';expected='value = 2\n'
        sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
        (call/'input-source.py').write_bytes(original.encode())
        (call/'output-source.py').write_bytes(output.encode())
        (call/'source-provenance.json').write_text(json.dumps(dict(candidate_existed=True,
            input_sha256=sha(original),output_sha256=sha(output))))
        tools={'change':dict(name='Edit',input=dict(file_path=str(workspace/'candidate.py'),
             old_string=original,new_string=expected))}
        return module.replay_candidate(call,workspace,tools,{'change':dict(is_error=False)})

    def test_recorded_model_edit_replays_exact_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=self.check(Path(tmp),'value = 2\n')
            self.assertEqual(row['status'],'verified')

    def test_unexplained_output_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):self.check(Path(tmp),'external_patch = 3\n')

    def test_context_must_match_input_method_and_actual_knowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            call=Path(tmp)
            (call/'method-manifest.json').write_bytes(b'{"files":[]}')
            method=module.sha(call/'method-manifest.json')
            (call/'source-provenance.json').write_text(json.dumps(dict(input_sha256='current')))
            (call/'initial-knowledge.json').write_text(json.dumps(dict(mechanisms=[dict(id='current')])))
            correct=dict(source_sha256='current',method_manifest_sha256=method,mechanisms=[dict(id='current')])
            for field,bad in [('source_sha256','old'),('method_manifest_sha256','old'),('mechanisms',[dict(id='old')])]:
                (call/'initial-context.json').write_text(json.dumps(dict(correct,**{field:bad})))
                with self.assertRaises(AssertionError):module.initial_context_identity(call)
            (call/'initial-context.json').write_text(json.dumps(correct))
            self.assertEqual(module.initial_context_identity(call)['status'],'verified')

    def test_initial_file_may_only_be_exact_upstream_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Path(tmp);starter=hashlib.sha256(b'upstream').hexdigest()
            generated=b'model-generated';final=hashlib.sha256(generated).hexdigest()
            (workspace/'candidate.py').write_bytes(generated)
            marker=workspace/'candidate-initialization.json'
            initialization=dict(kind='upstream_bytes_copy',source_sha256=starter,candidate_sha256=starter,implementation_changed=False)
            marker.write_text(json.dumps(initialization))
            rows=[dict(candidate_replay=dict(candidate_existed=True,input_sha256=starter,output_sha256=final))]
            module.frozen_source_continuity(rows,starter,workspace)
            marker.write_text(json.dumps(dict(initialization,candidate_sha256='different_source')))
            with self.assertRaises(AssertionError):module.frozen_source_continuity(rows,starter,workspace)
            marker.write_text(json.dumps(initialization))
            changed=[rows[0],dict(candidate_replay=dict(candidate_existed=True,input_sha256='external_patch',output_sha256=final))]
            with self.assertRaises(AssertionError):module.frozen_source_continuity(changed,starter,workspace)


if __name__=='__main__':unittest.main()
