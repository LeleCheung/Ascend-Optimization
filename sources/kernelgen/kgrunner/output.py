"""CC output parsing: session_id, error detection, structured output extraction."""

import json
import re
from pathlib import Path


def extract_session_id(jsonl_path: Path | str) -> str | None:
    """Extract session_id from the init event in a CC stream-json log."""
    try:
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "system" and event.get("subtype") == "init":
                    return event.get("session_id")
    except Exception:
        pass
    return None


_API_ERROR_PATTERNS = [
    "API Error",
    "Unexpected EOF",
    "connection reset",
    "ECONNRESET",
]


def is_api_stream_error(jsonl_path: Path | str) -> bool:
    """Check if a CC run was interrupted by an API streaming error (resumable)."""
    result_event = _get_result_event(jsonl_path)
    if not result_event:
        return False
    # 403/401 are auth errors, not resumable
    if result_event.get("api_error_status") is not None:
        return False
    result_text = result_event.get("result", "")
    if not any(pat in result_text for pat in _API_ERROR_PATTERNS):
        return False
    return extract_session_id(jsonl_path) is not None


def get_result_text(jsonl_path: Path | str) -> str:
    """Extract the result text from the last result event."""
    return _get_result_event(jsonl_path).get("result", "")


def parse_structured_output(
    jsonl_path: Path | str, required_fields: list[str] | None = None
) -> dict | None:
    """Extract structured JSON output from CC result text using brace-counting."""
    text = get_result_text(jsonl_path)
    if not text:
        return None

    # Try extracting from code blocks first
    code_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        block_text = code_match.group(1).strip()
        idx = block_text.find("{")
        if idx >= 0:
            obj_str = _extract_json_object(block_text, idx)
            if obj_str:
                result = _try_parse(obj_str, required_fields)
                if result is not None:
                    return result

    # Fallback: find any JSON object in the text
    idx = text.find("{")
    while idx >= 0:
        obj_str = _extract_json_object(text, idx)
        if obj_str:
            result = _try_parse(obj_str, required_fields)
            if result is not None:
                return result
            idx = text.find("{", idx + 1)
        else:
            idx = text.find("{", idx + 1)

    return None


def extract_text_content(jsonl_path: Path | str) -> str:
    """Extract all text content from assistant messages and result."""
    full_text = ""
    try:
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "assistant":
                    content = msg.get("message", {}).get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            full_text += block.get("text", "")
                elif msg.get("type") == "result":
                    result_text = msg.get("result", "")
                    if result_text:
                        full_text += result_text
    except Exception:
        pass
    return full_text


def extract_code_block(jsonl_path: Path | str, language: str = "python") -> str | None:
    """Extract the last code block of the given language from CC output."""
    text = extract_text_content(jsonl_path)
    if not text:
        return None
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_pattern(jsonl_path: Path | str, pattern: str) -> str | None:
    """Extract first regex match from CC output text."""
    text = extract_text_content(jsonl_path)
    if not text:
        return None
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _get_result_event(jsonl_path: Path | str) -> dict:
    try:
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    return event
    except Exception:
        pass
    return {}


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a complete JSON object using brace counting."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_parse(json_str: str, required_fields: list[str] | None) -> dict | None:
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if required_fields:
        if not all(f in obj for f in required_fields):
            return None
    return obj
