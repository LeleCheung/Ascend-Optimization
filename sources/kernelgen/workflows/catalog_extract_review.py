"""Read-only, content-bound review of one extraction attempt."""

import json
from pathlib import Path

from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
from kernelgen.data._atomic import atomic_write_json
from kernelgen.workflows.optimization.artifacts import catalog_identity, file_digest


class CatalogReviewRequired(RuntimeError):
    """The bounded loop exhausted its budget without an accepted Catalog."""


class CatalogExtractionBlocked(RuntimeError):
    """A persisted, evidence-backed Agent report stopped this extraction."""

    def __init__(self, report_path):
        self.report_path = Path(report_path)
        super().__init__(f"Catalog extraction blocked; evidence: {self.report_path}")


def review_link_path(catalog):
    catalog = Path(catalog)
    return catalog.with_name(catalog.name + '.review.json')


def publish_review_link(catalog, operator, review_path):
    """Keep the original report authoritative; never put metadata in the bundle."""
    review_path = Path(review_path).resolve()
    atomic_write_json(review_link_path(catalog), dict(
        schema_version=1, operator=operator, review_path=str(review_path),
        review_sha256=file_digest(review_path)))


def reusable_review(catalog, operator):
    """Validate a trusted local review record, not a signature from an external party.

    Missing, stale or malformed records trigger ordinary semantic review. All
    original evidence must remain available and unchanged, including source files.
    """
    try:
        link = json.loads(review_link_path(catalog).read_text())
        if link['schema_version'] != 1 or link['operator'] != operator:
            return None
        path = Path(link['review_path'])
        if file_digest(path) != link['review_sha256']:
            return None
        report = json.loads(path.read_text())
        parsed = ReviewOutput.model_validate({k: report[k] for k in ('summary', 'findings', 'reviewed_files', 'blocker') if k in report})
        if (report['accepted'] is not True or report['missing_evidence']
                or parsed.blocks_catalog()
                or report['subject_sha256'] != catalog_identity(catalog, operator)):
            return None
        evidence = report['evidence_sha256']
        if not isinstance(evidence, dict) or not evidence:
            return None
        read = {str(Path(p).resolve()) for p in parsed.reviewed_files}
        if any(str(Path(p).resolve()) not in read or file_digest(Path(p)) != digest
               for p, digest in evidence.items()):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def review_attempt(operator, catalog, evidence, workspace, runtime_factory):
    workspace.mkdir(parents=True, exist_ok=True)
    files = list(dict.fromkeys([*(p.resolve() for p in catalog.rglob('*') if p.is_file()),
                               *(Path(p).resolve() for p in evidence)]))
    before = {str(p): file_digest(p) for p in files}
    subject = catalog_identity(catalog, operator)
    report = ArtifactReviewerAgent().run({
        'kind':'catalog','operator':operator,'subject_sha256':subject,
        'evidence_paths':[str(p) for p in files],
    }, runtime_factory(str(workspace)))
    if before != {str(p): file_digest(p) for p in files} or subject != catalog_identity(catalog, operator):
        raise ValueError('review changed the Catalog or source evidence')
    missing = sorted(set(before)-{str(Path(p).resolve()) for p in report.reviewed_files})
    if report.blocker is not None:
        report.blocker.validate_evidence(before)
    accepted = not missing and not report.blocks_catalog()
    result = dict(subject_sha256=subject,accepted=accepted,missing_evidence=missing,
                  evidence_sha256=before,**report.model_dump(mode='json'))
    path = workspace/'review.json'
    atomic_write_json(path,result)
    if report.blocker is not None and not missing:
        raise CatalogExtractionBlocked(path)
    feedback = ('Revise the previous extraction using the following review evidence. '
        'Return a complete replacement matching the original output contract. '
        'Do not edit pinned source/tests, drop source cases, relax tolerances, or use target '
        'limitations to change the common Catalog. If a finding is mistaken, preserve the '
        'source-faithful behavior and cite that evidence for the next independent review.\n'
        'Only repair conversion defects. source_quality and target_capability findings '
        'are records for pytest review and target readiness, not instructions to alter the Catalog.\n'
        + json.dumps(result,ensure_ascii=False))
    return accepted, path, feedback
