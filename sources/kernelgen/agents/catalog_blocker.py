"""Evidence-backed proposal to stop an unrepresentable Catalog extraction."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CatalogBlocker(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["protocol", "extraction_environment"]
    source_requirement: str = Field(min_length=1)
    evidence_paths: list[str] = Field(min_length=1)
    missing_contract: str = Field(min_length=1)
    resolution: str = Field(min_length=1)

    def validate_evidence(self, evidence) -> None:
        """Check references, not the truth of the model's technical conclusion."""
        allowed = {Path(p).resolve() for p in evidence}
        if any(not Path(p).is_absolute() or Path(p).resolve() not in allowed
               for p in self.evidence_paths):
            raise ValueError("Catalog blocker must cite the supplied source evidence paths")
