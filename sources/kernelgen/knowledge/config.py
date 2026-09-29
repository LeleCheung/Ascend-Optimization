"""Knowledge subsystem configuration.

Omitting ``KnowledgeConfig`` disables V1 for a workflow. ``read_write_v1`` is
the authoritative publication mode; ``read_only_v1`` permits retrieval while
keeping the Catalog immutable. Runtime Candidate review is independently
controlled and defaults to fail-closed ``off``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class KnowledgeMode(str, Enum):
    READ_WRITE_V1 = "read_write_v1"
    READ_ONLY_V1 = "read_only_v1"


class KnowledgeReviewerMode(str, Enum):
    """How runtime Candidates are semantically reviewed at epoch publication."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class KnowledgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: KnowledgeMode = KnowledgeMode.READ_WRITE_V1
    reviewer_mode: KnowledgeReviewerMode = KnowledgeReviewerMode.OFF
    catalog_root: Path
    run_archive_root: Path | None = None
    derived_root: Path | None = None

    @model_validator(mode="after")
    def validate_access_mode(self) -> "KnowledgeConfig":
        if (
            self.mode == KnowledgeMode.READ_ONLY_V1
            and self.reviewer_mode != KnowledgeReviewerMode.OFF
        ):
            raise ValueError(
                "read_only_v1 requires knowledge reviewer mode=off"
            )
        if (
            self.mode == KnowledgeMode.READ_ONLY_V1
            and self.derived_root is not None
        ):
            catalog = self.catalog_root.resolve()
            derived = self.derived_root.resolve()
            if derived == catalog or catalog in derived.parents:
                raise ValueError(
                    "read_only_v1 derived_root must be outside catalog_root"
                )
        return self

    @property
    def reads_v1(self) -> bool:
        return self.mode in {
            KnowledgeMode.READ_WRITE_V1,
            KnowledgeMode.READ_ONLY_V1,
        }

    @property
    def writes_v1(self) -> bool:
        return self.mode == KnowledgeMode.READ_WRITE_V1

    @property
    def resolved_derived_root(self) -> Path:
        return self.derived_root or (self.catalog_root.resolve() / ".derived")

    @property
    def index_rebuild_enabled(self) -> bool:
        return self.writes_v1 or self.derived_root is not None

    @property
    def resolved_run_archive_root(self) -> Path:
        return self.run_archive_root or (
            self.catalog_root.resolve().parent / "run-archive"
        )
