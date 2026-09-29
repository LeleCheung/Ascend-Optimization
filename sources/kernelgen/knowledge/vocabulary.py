"""Catalog-owned vocabulary normalization and audit helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from kernelgen.knowledge.taxonomy import (
    OperatorTaxonomy,
    normalize_taxonomy_term,
)


_CATEGORIES = ("techniques", "symptoms")
_CANONICAL_FILES = {
    "techniques": "canonical_techniques.yaml",
    "symptoms": "canonical_symptoms.yaml",
}
_UNIQUE_LIMITS = {"techniques": 30, "symptoms": 15}


def _key(value: str) -> str:
    parts = []
    pending_separator = False
    for character in value.strip().lower():
        if character.isalnum():
            if pending_separator and parts:
                parts.append("_")
            parts.append(character)
            pending_separator = False
        else:
            pending_separator = True
    return "".join(parts)


def canonical_failure_symptom(status: str) -> str:
    """Map evaluator failure statuses onto Catalog symptom vocabulary."""

    normalized = status.strip().upper()
    if normalized == "COMPILE_ERROR":
        return "compile_error"
    if normalized == "INCORRECT_NUMERICAL":
        return "precision_error"
    if normalized == "PARTIAL_PASS" or normalized.startswith("INCORRECT_"):
        return "correctness_failure"
    if normalized in {"RUNTIME_ERROR", "TIMEOUT"}:
        return "runtime_failure"
    if normalized in {"OUT_OF_MEMORY", "OOM", "RESOURCE_EXHAUSTED"}:
        return "resource_exhaustion"
    return status


def _read_mapping(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return payload


def _read_canonical_terms(path: Path) -> tuple[str, ...]:
    payload = _read_mapping(path)
    if not payload:
        return ()
    terms = payload.get("terms")
    if not isinstance(terms, list) or any(
        not isinstance(item, str) or not item.strip() for item in terms
    ):
        raise ValueError(f"{path.name}.terms must contain non-empty strings")
    normalized = [_key(item) for item in terms]
    if any(not item for item in normalized):
        raise ValueError(f"{path.name}.terms contains an empty normalized term")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path.name}.terms contains duplicate terms")
    return tuple(normalized)


@dataclass(frozen=True)
class Vocabulary:
    """Canonical terms and exact aliases loaded from one Catalog."""

    families: dict[str, tuple[str, ...]] = field(default_factory=dict)
    owners: dict[str, tuple[str, str]] = field(default_factory=dict)
    canonicals: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_catalog(cls, catalog_root: Path) -> "Vocabulary":
        root = Path(catalog_root)
        canonicals = {
            category: _read_canonical_terms(root / filename)
            for category, filename in _CANONICAL_FILES.items()
        }
        payload = _read_mapping(root / "aliases.yaml")

        families: dict[str, tuple[str, ...]] = {}
        owners: dict[str, tuple[str, str]] = {}
        derived_canonicals: dict[str, list[str]] = {
            category: list(canonicals[category]) for category in _CATEGORIES
        }
        for category in _CATEGORIES:
            values = payload.get(category, {})
            if not isinstance(values, dict):
                raise ValueError(f"aliases.{category} must contain a mapping")
            allowed = set(canonicals[category])
            for raw_canonical, aliases in values.items():
                if not isinstance(raw_canonical, str) or not raw_canonical.strip():
                    raise ValueError(f"aliases.{category} has an empty canonical term")
                canonical = _key(raw_canonical)
                if allowed and canonical not in allowed:
                    raise ValueError(
                        f"aliases.{category}.{raw_canonical} is not listed in "
                        f"{_CANONICAL_FILES[category]}"
                    )
                if not isinstance(aliases, list) or any(
                    not isinstance(item, str) or not item.strip()
                    for item in aliases
                ):
                    raise ValueError(
                        f"aliases.{category}.{raw_canonical} must contain strings"
                    )
                if canonical not in derived_canonicals[category]:
                    derived_canonicals[category].append(canonical)
                terms = tuple(dict.fromkeys([canonical, *aliases]))
                owner = (category, canonical)
                for term in terms:
                    normalized = _key(term)
                    previous = owners.get(normalized)
                    if previous is not None and previous != owner:
                        raise ValueError(
                            f"ambiguous vocabulary term {term!r}: "
                            f"{previous[0]}.{previous[1]} and {category}.{canonical}"
                        )
                    owners[normalized] = owner
                    families[normalized] = terms

        # A canonical term remains searchable and valid even before aliases are
        # discovered for it from real Source queries.
        for category in _CATEGORIES:
            for canonical in derived_canonicals[category]:
                owner = (category, canonical)
                previous = owners.get(canonical)
                if previous is not None and previous != owner:
                    raise ValueError(
                        f"ambiguous canonical term {canonical!r}: "
                        f"{previous[0]}.{previous[1]} and {category}.{canonical}"
                    )
                owners.setdefault(canonical, owner)
                families.setdefault(canonical, (canonical,))

        taxonomy = OperatorTaxonomy.from_catalog(root)
        if taxonomy is not None:
            for normalized, terms in taxonomy.alias_families.items():
                owner = ("taxonomy", normalize_taxonomy_term(terms[0]))
                previous = owners.get(normalized)
                if previous is not None and previous != owner:
                    raise ValueError(
                        f"ambiguous vocabulary term {normalized!r}: "
                        f"{previous[0]}.{previous[1]} and "
                        f"taxonomy.{owner[1]}"
                    )
                existing = families.get(normalized)
                if existing is not None and existing != terms:
                    raise ValueError(
                        f"ambiguous vocabulary expansion for {normalized!r}"
                    )
                owners[normalized] = owner
                families[normalized] = terms

        return cls(
            families=families,
            owners=owners,
            canonicals={
                category: tuple(derived_canonicals[category])
                for category in _CATEGORIES
            },
        )

    def canonical_terms(self, category: str) -> list[str]:
        self._validate_category(category)
        return list(self.canonicals.get(category, ()))

    def aliases_for(self, category: str, canonical: str) -> list[str]:
        self._validate_category(category)
        normalized = _key(canonical)
        owner = self.owners.get(normalized)
        if owner != (category, normalized):
            return []
        return list(self.families.get(normalized, (normalized,)))[1:]

    def canonicalize(self, category: str, value: str) -> str | None:
        self._validate_category(category)
        owner = self.owners.get(_key(value))
        if owner is None or owner[0] != category:
            return None
        return owner[1]

    def normalize_values(
        self,
        category: str,
        values: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        """Canonicalize known values and preserve unknown values with warnings."""

        self._validate_category(category)
        configured = bool(self.canonicals.get(category)) or any(
            owner_category == category
            for owner_category, _ in self.owners.values()
        )
        if not configured:
            preserved = [str(value).strip() for value in values if str(value).strip()]
            return list(dict.fromkeys(preserved)), []

        normalized = []
        unknown = []
        for value in values:
            stripped = str(value).strip()
            if not stripped:
                continue
            canonical = self.canonicalize(category, stripped)
            if canonical is None:
                normalized.append(stripped)
                unknown.append(stripped)
            else:
                normalized.append(canonical)
        return list(dict.fromkeys(normalized)), list(dict.fromkeys(unknown))

    def expand(self, query: str) -> list[str]:
        """Return exact alias expansions while preserving unknown queries."""

        normalized_query = _key(query)
        direct = self.families.get(normalized_query)
        if direct is not None:
            return list(direct)

        expanded = [query]
        seen_families: set[tuple[str, ...]] = set()
        for term, family in self.families.items():
            if family in seen_families:
                continue
            if _contains_term(normalized_query, term):
                expanded.extend(family)
                seen_families.add(family)
        return list(dict.fromkeys(expanded))

    @staticmethod
    def _validate_category(category: str) -> None:
        if category not in _CATEGORIES:
            raise ValueError(f"unsupported vocabulary category: {category}")


def _contains_term(normalized_query: str, term: str) -> bool:
    if not term:
        return False
    if any("\u4e00" <= character <= "\u9fff" for character in term):
        return term in normalized_query
    return f"_{term}_" in f"_{normalized_query}_"


def concept_vocabulary_warnings(
    concepts: Iterable[object],
    vocabulary: Vocabulary,
) -> list[str]:
    """Return non-blocking unknown-value and cardinality warnings."""

    warnings = []
    values_by_category = {category: set() for category in _CATEGORIES}
    fields = {"techniques": "techniques", "symptoms": "symptoms"}
    for concept in concepts:
        retrieval = getattr(concept, "retrieval", None)
        if retrieval is None:
            continue
        for category, field_name in fields.items():
            if not vocabulary.canonical_terms(category):
                continue
            for value in getattr(retrieval, field_name, []):
                values_by_category[category].add(value)
                if vocabulary.canonicalize(category, value) is None:
                    warnings.append(
                        f"{getattr(concept, 'id', '<unknown>')}: unknown "
                        f"{category[:-1]} {value!r}"
                    )
    for category, values in values_by_category.items():
        limit = _UNIQUE_LIMITS[category]
        if len(values) > limit:
            warnings.append(
                f"catalog has {len(values)} unique {category}; recommended maximum "
                f"is {limit}"
            )
    return sorted(set(warnings))
