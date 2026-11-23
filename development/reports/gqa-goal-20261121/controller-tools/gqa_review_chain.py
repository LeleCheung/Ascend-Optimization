"""Review a model-authored BF16 PV accumulator-chain candidate without editing it."""
import argparse
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

from gqa_review_bindings import review
from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    common = review(run, args.invocation, ('.ttir',))
    before, after = [ast.parse((call / name).read_text()) for name in ['input-source.py', 'output-source.py']]
    get_kernel = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gqa_fwd_kernel')
    original, changed = get_kernel(before), copy.deepcopy(get_kernel(after))
    assert changed.args.args[-1].arg == 'CHAIN_LO'
    changed.args.args.pop()
    branches = [n for n in ast.walk(changed) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'CHAIN_LO']
    assert len(branches) == 1
    branch = branches[0]
    class OriginalFP16Path(ast.NodeTransformer):
        def visit_If(self, node):
            return node.orelse if node is branch else self.generic_visit(node)
    restored = OriginalFP16Path().visit(changed)
    assert ast.dump(original, include_attributes=False) == ast.dump(restored, include_attributes=False)
    config = read(run / 'config.json')
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.gate import inspect_text, admit
    source = call / 'output-source.py'
    screening = inspect_text(source.read_text())
    assert screening['static_passed']
    record = dict(status='reviewed_strict_triton', source_sha256=sha(source),
        all_branches_reviewed=True, dependencies_reviewed=True,
        scope='Model-authored CHAIN_LO selected only by input dtype metadata. Host allocates a fresh output and launches '
              'the same grid with unchanged strides. Kernel pointer arithmetic, bounds, softmax and original FP16 path '
              'are unchanged. BF16 uses two tl.dot calls with an accumulator dependency and prescaled residual operand. '
              'No external compute, case-id dispatch, cached outputs or input construction.',
        fp16_kernel_ast_after_constexpr_elimination='identical',
        numerical_risk='BF16 residual prescaling and seeded second dot change FP32 accumulation grouping and rounding. '
                       'Power-of-two scaling alone does not establish bitwise equivalence. Official four-case checker '
                       'is required; no equivalence claim is accepted.',
        lowering_prediction='A tt.dot chain may still lower to separate L0C transfers; inspect actual products and replay.',
        evidence_scope='Model successfully read current simulator, profiling, source and TTIR before decision. '
                       'No preedit model BCMLIR read logged; TTIR identifies the selected source-level dot mechanism. '
                       'Codex independently inspected BCMLIR three-matmul structure; contribution kept separate.')
    (run / 'source-reviews').mkdir(exist_ok=True)
    save(run / 'source-reviews' / (sha(source) + '.json'), record)
    admit(source.read_text(), record)
    common.update(status='preedit_reads_and_semantics_approved',
        semantic_change='BF16 residual prescaling and PV accumulator chain; FP16 original AST preserved.',
        numerical_risk=record['numerical_risk'], lowering_prediction=record['lowering_prediction'],
        evidence_scope=record['evidence_scope'], reviewed_at=datetime.now(timezone.utc).isoformat())
    save(call / 'host-review.json', common)
    print(dict(status=common['status'], source_sha256=sha(source), fp16_original_ast=True))


if __name__ == '__main__':
    main()
