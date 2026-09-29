"""Canonical operator taxonomy owned by one knowledge Catalog.

The taxonomy is optional for legacy Catalogs. Once a Catalog provides
``operator_taxonomy.yaml`` it becomes authoritative: inactive taxonomies and
unmapped Definitions fail closed instead of falling back to source-text
heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class OperatorTaxonomyError(ValueError):
    """Raised when a Catalog taxonomy is malformed or cannot be activated."""


class UnmappedOperatorDefinitionError(OperatorTaxonomyError):
    """Raised when an active taxonomy has no mapping for one Definition."""


def normalize_taxonomy_term(value: str) -> str:
    parts: list[str] = []
    pending_separator = False
    for character in str(value).strip().casefold():
        if character.isalnum():
            if pending_separator and parts:
                parts.append("_")
            parts.append(character)
            pending_separator = False
        else:
            pending_separator = True
    return "".join(parts)


@dataclass(frozen=True)
class DefinitionMapping:
    family: str
    motifs: tuple[str, ...]
    dataflow: tuple[str, ...]
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    workload_features: dict[str, Any]


@dataclass(frozen=True)
class OperatorTaxonomy:
    schema_version: str
    taxonomy_version: str
    families: frozenset[str]
    motifs: frozenset[str]
    dataflows: frozenset[str]
    definitions: dict[str, DefinitionMapping]
    alias_families: dict[str, tuple[str, ...]]

    @classmethod
    def from_catalog(
        cls,
        catalog_root: Path,
        *,
        required: bool = False,
        allow_preview: bool = False,
    ) -> "OperatorTaxonomy | None":
        path = Path(catalog_root) / "operator_taxonomy.yaml"
        if not path.is_file():
            if required:
                raise OperatorTaxonomyError(
                    f"operator taxonomy is required but missing: {path}"
                )
            return None
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise OperatorTaxonomyError(
                f"unreadable operator taxonomy {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise OperatorTaxonomyError("operator taxonomy root must be a mapping")
        return cls.from_mapping(payload, allow_preview=allow_preview)

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        allow_preview: bool = False,
    ) -> "OperatorTaxonomy":
        if payload.get("schema_version") != "1.1":
            raise OperatorTaxonomyError(
                "operator taxonomy schema_version must be '1.1'"
            )
        taxonomy_version = _non_empty_string(
            payload.get("taxonomy_version"), "taxonomy_version"
        )
        activation = _mapping(payload.get("activation"), "activation")
        status = activation.get("status")
        compatible = activation.get("client_compatible")
        if not allow_preview and (status != "active" or compatible is not True):
            raise OperatorTaxonomyError(
                "operator taxonomy is not active and client-compatible"
            )

        scope_semantics = _mapping(
            payload.get("scope_semantics"), "scope_semantics"
        )
        expected_semantics = {
            "definition_ids": "exact_any_overlap",
            "families": "any_overlap",
            "motifs": "concept_required_subset_of_query",
            "dataflows": "any_overlap",
            "dtypes": "any_overlap",
            "layouts": "any_overlap",
            "aliases": "exact_token",
        }
        for field, expected in expected_semantics.items():
            if scope_semantics.get(field) != expected:
                raise OperatorTaxonomyError(
                    f"scope_semantics.{field} must be {expected!r}"
                )

        sections = {
            name: _mapping(payload.get(name), name)
            for name in ("families", "motifs", "dataflows")
        }
        for name, section in sections.items():
            if not section:
                raise OperatorTaxonomyError(f"{name} must not be empty")

        families = frozenset(sections["families"])
        motifs = frozenset(sections["motifs"])
        dataflows = frozenset(sections["dataflows"])
        alias_families = _alias_families(sections)

        dataflow_specs: dict[str, tuple[str, frozenset[str]]] = {}
        for dataflow_id, raw in sections["dataflows"].items():
            spec = _mapping(raw, f"dataflows.{dataflow_id}")
            family = _non_empty_string(
                spec.get("family"), f"dataflows.{dataflow_id}.family"
            )
            required_motifs = frozenset(
                _string_list(
                    spec.get("required_motifs"),
                    f"dataflows.{dataflow_id}.required_motifs",
                    required=True,
                )
            )
            if family not in families:
                raise OperatorTaxonomyError(
                    f"dataflows.{dataflow_id} references unknown family {family!r}"
                )
            unknown = required_motifs - motifs
            if unknown:
                raise OperatorTaxonomyError(
                    f"dataflows.{dataflow_id} references unknown motifs "
                    f"{sorted(unknown)}"
                )
            dataflow_specs[dataflow_id] = (family, required_motifs)

        raw_definitions = _mapping(payload.get("definitions"), "definitions")
        if not raw_definitions:
            raise OperatorTaxonomyError("definitions must not be empty")
        definitions: dict[str, DefinitionMapping] = {}
        for definition_id, raw in raw_definitions.items():
            spec = _mapping(raw, f"definitions.{definition_id}")
            family = _non_empty_string(
                spec.get("family"), f"definitions.{definition_id}.family"
            )
            definition_motifs = tuple(
                _string_list(
                    spec.get("motifs"),
                    f"definitions.{definition_id}.motifs",
                    required=True,
                )
            )
            definition_dataflow = tuple(
                _string_list(
                    spec.get("dataflow"),
                    f"definitions.{definition_id}.dataflow",
                    required=True,
                )
            )
            dtypes = tuple(
                _string_list(
                    spec.get("dtypes"),
                    f"definitions.{definition_id}.dtypes",
                    required=True,
                )
            )
            layouts = tuple(
                _string_list(
                    spec.get("layouts"),
                    f"definitions.{definition_id}.layouts",
                    required=True,
                )
            )
            workload_features = _mapping(
                spec.get("workload_features"),
                f"definitions.{definition_id}.workload_features",
            )
            if family not in families:
                raise OperatorTaxonomyError(
                    f"definitions.{definition_id} references unknown family {family!r}"
                )
            unknown_motifs = set(definition_motifs) - motifs
            unknown_dataflows = set(definition_dataflow) - dataflows
            if unknown_motifs or unknown_dataflows:
                raise OperatorTaxonomyError(
                    f"definitions.{definition_id} references unknown taxonomy terms "
                    f"motifs={sorted(unknown_motifs)} "
                    f"dataflows={sorted(unknown_dataflows)}"
                )
            for dataflow_id in definition_dataflow:
                dataflow_family, required_motifs = dataflow_specs[dataflow_id]
                if dataflow_family != family:
                    raise OperatorTaxonomyError(
                        f"definitions.{definition_id} family differs from "
                        f"dataflow {dataflow_id}"
                    )
                missing = required_motifs - set(definition_motifs)
                if missing:
                    raise OperatorTaxonomyError(
                        f"definitions.{definition_id} is missing motifs required "
                        f"by {dataflow_id}: {sorted(missing)}"
                    )
            definitions[definition_id] = DefinitionMapping(
                family=family,
                motifs=definition_motifs,
                dataflow=definition_dataflow,
                dtypes=dtypes,
                layouts=layouts,
                workload_features=dict(workload_features),
            )

        return cls(
            schema_version="1.1",
            taxonomy_version=taxonomy_version,
            families=families,
            motifs=motifs,
            dataflows=dataflows,
            definitions=definitions,
            alias_families=alias_families,
        )

    def mapping_for(self, definition_id: str) -> DefinitionMapping:
        mapping = self.definitions.get(definition_id)
        if mapping is None:
            raise UnmappedOperatorDefinitionError(
                f"Definition {definition_id!r} is unmapped in active taxonomy "
                f"{self.taxonomy_version!r}"
            )
        return mapping

    def validate_operator_scope(self, operator: Any, *, label: str) -> None:
        unknown_definitions = set(operator.definition_ids) - set(self.definitions)
        unknown_families = set(operator.op_types) - self.families
        unknown_motifs = set(operator.motifs) - self.motifs
        unknown_dataflows = set(operator.dataflow) - self.dataflows
        if any(
            (unknown_definitions, unknown_families, unknown_motifs, unknown_dataflows)
        ):
            raise OperatorTaxonomyError(
                f"{label} contains unknown canonical operator terms: "
                f"definitions={sorted(unknown_definitions)} "
                f"families={sorted(unknown_families)} "
                f"motifs={sorted(unknown_motifs)} "
                f"dataflows={sorted(unknown_dataflows)}"
            )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OperatorTaxonomyError(f"{label} must be a mapping")
    return value


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperatorTaxonomyError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, label: str, *, required: bool = False) -> list[str]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise OperatorTaxonomyError(f"{label} must contain non-empty strings")
    output = [item.strip() for item in value]
    if required and not output:
        raise OperatorTaxonomyError(f"{label} must not be empty")
    if len(output) != len(set(output)):
        raise OperatorTaxonomyError(f"{label} contains duplicate values")
    return output


def _alias_families(
    sections: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    owners: dict[str, tuple[str, str]] = {}
    output: dict[str, tuple[str, ...]] = {}
    for section_name, section in sections.items():
        for canonical, raw in section.items():
            spec = _mapping(raw, f"{section_name}.{canonical}")
            aliases = _string_list(
                spec.get("aliases", []), f"{section_name}.{canonical}.aliases"
            )
            terms = tuple(dict.fromkeys((canonical, *aliases)))
            owner = (section_name, canonical)
            for term in terms:
                normalized = normalize_taxonomy_term(term)
                if not normalized:
                    raise OperatorTaxonomyError(
                        f"{section_name}.{canonical} contains an empty alias"
                    )
                previous = owners.get(normalized)
                if previous is not None and previous != owner:
                    raise OperatorTaxonomyError(
                        f"ambiguous taxonomy alias {term!r}: "
                        f"{previous[0]}.{previous[1]} and "
                        f"{section_name}.{canonical}"
                    )
                owners[normalized] = owner
                output[normalized] = terms
    return output
