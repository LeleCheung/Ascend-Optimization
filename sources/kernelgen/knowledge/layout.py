"""One owner for every workspace and catalog knowledge path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def safe_name(value: str) -> str:
    """Return a filesystem- and identifier-safe name without changing case."""

    normalized = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return normalized.strip("-") or "unknown"


@dataclass(frozen=True)
class KnowledgeLayout:
    workspace: Path

    @property
    def root(self) -> Path:
        return self.workspace / ".kernelgen" / "knowledge"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def context_dir(self) -> Path:
        return self.root / "context"

    @property
    def operator_signature(self) -> Path:
        return self.context_dir / "operator-signature.json"

    @property
    def target_context(self) -> Path:
        return self.context_dir / "target-context.json"

    @property
    def query_log(self) -> Path:
        return self.root / "query-log.jsonl"

    @property
    def retrieval_log(self) -> Path:
        return self.root / "retrieval-log.jsonl"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates.jsonl"

    @property
    def publish_result(self) -> Path:
        return self.root / "publish-result.json"


@dataclass(frozen=True)
class DerivedLayout:
    """Disposable indexes, optionally stored outside an immutable Catalog."""

    root: Path

    @property
    def index(self) -> Path:
        return self.root / "knowledge.db"

    @property
    def source_index(self) -> Path:
        return self.root / "sources.db"

    @property
    def round_index(self) -> Path:
        return self.root / "rounds.db"


@dataclass(frozen=True)
class CatalogLayout:
    root: Path

    @property
    def concepts(self) -> Path:
        return self.root / "concepts"

    @property
    def observations(self) -> Path:
        return self.root / "observations" / "by_run"

    @property
    def solutions(self) -> Path:
        """Current best kernel slots, grouped by exact target and definition."""
        return self.root / "solutions"

    @property
    def round_index(self) -> Path:
        return DerivedLayout(self.derived).round_index

    @property
    def source_packages(self) -> Path:
        return self.root / "sources" / "packages"

    @property
    def source_content(self) -> Path:
        return self.root / "sources" / "content"

    @property
    def derived(self) -> Path:
        return self.root / ".derived"

    @property
    def index(self) -> Path:
        return DerivedLayout(self.derived).index

    @property
    def source_index(self) -> Path:
        return DerivedLayout(self.derived).source_index

    @property
    def publish_lock(self) -> Path:
        return self.root / ".publish.lock"

    @property
    def publish_transaction_lock(self) -> Path:
        """Serialize archive, indexes, catalog publication, and Git commit."""
        return (
            self.root.parent
            / f".{self.root.name}.publish-transaction.lock"
        )

    @property
    def publish_audit(self) -> Path:
        return self.root / "publish" / "audit"

    @property
    def publish_reviews(self) -> Path:
        return self.root / "publish" / "reviews"
