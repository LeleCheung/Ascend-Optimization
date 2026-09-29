"""Read bounded, untrusted design evidence without importing or executing it."""

from pathlib import Path


def load_reference_code(path: Path | None, prompt_path: Path | None) -> dict[str, str]:
    if prompt_path is not None and path is None:
        raise ValueError("reference_code_prompt_path requires reference_code_path")
    return {
        "reference_code_source": _load_reference_text(path, "reference_code_path"),
        "reference_code_prompt": _load_reference_text(prompt_path, "reference_code_prompt_path"),
    }


def _load_reference_text(path: Path | None, field_name: str) -> str:
    if path is None:
        return ""
    max_bytes = 1_000_000
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{field_name} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a regular file: {path}")
    if resolved.stat().st_size > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte prompt limit: {path}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be UTF-8 text: {path}") from exc
    if not text.strip():
        raise ValueError(f"{field_name} is empty: {path}")
    return text
