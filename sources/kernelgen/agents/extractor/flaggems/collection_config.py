"""Workspace-local extraction settings, independent of KGS instance settings."""

import re

from pydantic import BaseModel, ConfigDict, field_validator

from kernelgen.framework.local_state import atomic_write_json, file_lock, read_json, state_home


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str
    container: str
    python: str

    @field_validator("host", "container", "python")
    @classmethod
    def validate_setting(cls, value, info):
        return validate_setting(info.field_name, value)


def validate_setting(key: str, value: str) -> str:
    if key in {"host", "container"}:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"extract.{key} must be an SSH Host alias or Docker name, without options")
    elif key == "python":
        if not value.startswith("/") or any(c in value for c in "\n\r\0"):
            raise ValueError("extract.python must be an absolute container interpreter path")
    else:
        raise ValueError("supported extraction keys: extract.host, extract.container, extract.python")
    return value


def set_extraction_setting(key: str, value: str) -> None:
    value = validate_setting(key, value)
    root = state_home()
    with file_lock(root / "config.lock"):
        raw = read_json(root / "config.json", {})
        raw.setdefault("extract", {})[key] = value
        atomic_write_json(root / "config.json", raw)


def load_extraction_config() -> ExtractionConfig:
    raw = read_json(state_home() / "config.json", {}).get("extract", {})
    missing = {"host", "container", "python"} - raw.keys()
    if missing:
        raise ValueError("configure collection with kg config set: " + ", ".join(f"extract.{key}" for key in sorted(missing)))
    return ExtractionConfig.model_validate(raw)
