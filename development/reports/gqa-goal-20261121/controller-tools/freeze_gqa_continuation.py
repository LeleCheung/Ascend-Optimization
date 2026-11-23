"""Freeze the existing accepted GQA candidate as a new research parent."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    previous=args.previous
    result=json.loads((previous/'with-simulator-v2/evaluation-round-02/results.json').read_bytes())
    source=previous/'with-simulator-v2/evaluation-round-02/source.py'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==result['source_sha256']
    assert all(c['candidate_correctness']['official_checker']['passed'] for c in result['cases'])
    args.output.mkdir(parents=True,exist_ok=False)
    files={}
    for name,path in [('parent.py',source),('contract.json',previous/'contract.json'),('parent-results.json',previous/'with-simulator-v2/evaluation-round-02/results.json')]:
        raw=path.read_bytes();(args.output/name).write_bytes(raw)
        files[name]={'origin':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    manifest={'schema':'hgkda.gqa-continuation/v1','status':'parent_frozen_no_new_measurement',
              'parent_source_sha256':sha,'files':files,'target_case':4,'validation_cases':[1,2,4,7],
              'objective':'Simulator evidence -> mechanism -> select Triton/IR/pass -> real-device validation',
              'required_before_edit':['evidence_hashes','provided_and_read_evidence','alternative_explanations','simulator_incremental_evidence','selected_layer_and_reason','prediction','validation_plan'],
              'required_after_edit':['parent_and_child_sha256','diff','actual_ir','compile_result','correctness','walltime','recapture','retain_or_revert','model_cost','codex_feedback'],
              'historical_limitations':['four-case subset, not full official suite','simulator partial windows','Codex review contribution']}
    (args.output/'continuation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__':main()
