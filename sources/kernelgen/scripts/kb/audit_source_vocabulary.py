"""Audit canonical vocabulary coverage against one Source Catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from kernelgen.data._atomic import atomic_write_text
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.sources import SourceSearcher
from kernelgen.knowledge.vocabulary import (
    Vocabulary,
    concept_vocabulary_warnings,
)


def audit_source_vocabulary(
    catalog_root: Path,
    *,
    max_results: int = 20,
) -> dict:
    root = Path(catalog_root).resolve()
    vocabulary = Vocabulary.from_catalog(root)
    searcher = SourceSearcher(root)
    family_results = []
    missing_families = []
    inconsistent_families = []

    for category in ("symptoms", "techniques"):
        for canonical in vocabulary.canonical_terms(category):
            aliases = vocabulary.aliases_for(category, canonical)
            terms = [canonical, *aliases]
            result = searcher.search(canonical, max_results=max_results)
            hits = [
                [
                    hit.source_package,
                    hit.revision,
                    hit.locator,
                ]
                for hit in result.hits
            ]
            expected_expansion = set(result.expanded_terms)
            consistent = all(
                set(vocabulary.expand(term)) == expected_expansion
                for term in terms
            )
            has_hits = bool(hits)
            item = {
                "category": category,
                "canonical": canonical,
                "aliases": aliases,
                "expanded_terms": result.expanded_terms,
                "hit_count": len(hits),
                "top_hits": [
                    {
                        "source_package": hit.source_package,
                        "path": hit.path,
                        "locator": hit.locator,
                        "score": hit.score,
                    }
                    for hit in result.hits[:5]
                ],
                "consistent": consistent,
            }
            family_results.append(item)
            if not has_hits:
                missing_families.append(f"{category}.{canonical}")
            if not consistent:
                inconsistent_families.append(f"{category}.{canonical}")

    catalog = FilesystemCatalog(root)
    concepts = catalog.iter_concepts()
    warnings = concept_vocabulary_warnings(concepts, vocabulary)
    warnings.extend(
        f"no Source hits for {item}" for item in missing_families
    )
    warnings.extend(
        f"alias results differ for {item}" for item in inconsistent_families
    )
    return {
        "schema_version": "1.0",
        "source_snapshot": catalog.snapshot(),
        "concept_count": len(concepts),
        "canonical_counts": {
            category: len(vocabulary.canonical_terms(category))
            for category in ("symptoms", "techniques")
        },
        "families": family_results,
        "missing_families": missing_families,
        "inconsistent_families": inconsistent_families,
        "audit_warnings": sorted(set(warnings)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    report = audit_source_vocabulary(
        arguments.catalog,
        max_results=arguments.max_results,
    )
    rendered = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    if arguments.output is not None:
        atomic_write_text(arguments.output, rendered)
    print(rendered, end="")
    return 1 if report["inconsistent_families"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
