# Language and provenance policy

Effective 2026-11-24, following the completed presentation phase.

## Maintained engineering material

Use English for identifiers, new paths, comments, docstrings, exceptions, CLI help,
generated diagnostic prose, model prompts, and configuration descriptions. Use UTF-8.
Preserve public keys, commands, and protocol identities.

Documentation may be English-only or bilingual English/Chinese. Bilingual material
provides an English counterpart, not just an English heading. Source names and
provenance paths are exact identifiers, not translatable prose.

## Immutable records

Do not translate frozen candidates, raw captures, contracts, model conversations,
formal results, or pinned upstream source in place. Their hashes and source mappings
must continue to resolve. This covers:

- `development/experiments/` and `development/dependencies/`;
- captured/generated artifacts and original evidence in `development/reports/`;
- hash-bound versions, contracts, acceptance resources, and evidence in `operators/`;
- raw source/capture examples in `hgkda/examples/`;
- explicitly archived historical source and documents.

These records retain their original language. Add English navigation beside them.
Translated working derivatives require separate paths, new identities, and links
to originals. Historical provenance paths retain their exact spelling.

## Validation

Language edits preserve computational behavior, validation requirements, and evidence
scope. Update tests asserting wording while retaining substantive checks. Run package,
distribution, and integrity checks. Record moves and update maintained references;
resolve raw historical references with maps instead of rewriting them.

## Automated scope

`development/manifests/english-language-scope.json` declares maintained roots/files,
immutable records, and six exact historical knowledge source identifiers. Run
`python -m development.tools.maintenance.audit_language` from the repository root.
The check rejects new Chinese prose in maintained source and English guides while
verifying retained knowledge path/hash bindings. It does not translate records.

The original example README is itself bound by the earlier immutable-file manifest;
its English working guide is `hgkda/docs/examples.md`. Earlier dated research notes
and delivered presentation documents remain historical records. Current entry-point
and runbook documentation is English.
