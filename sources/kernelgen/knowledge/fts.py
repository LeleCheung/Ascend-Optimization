"""Shared full-text query construction and rank fusion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_ENGLISH_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "of",
    "on",
    "should",
    "that",
    "the",
    "this",
    "to",
    "use",
    "what",
    "with",
}


def english_tokens(text: str) -> list[str]:
    """Return stable ASCII word tokens understood by the unicode61 index."""

    tokens: list[str] = []
    current: list[str] = []
    for character in str(text).casefold():
        if character.isascii() and character.isalnum():
            current.append(character)
        elif current:
            token = "".join(current)
            if len(token) >= 2 and token not in _ENGLISH_STOPWORDS:
                tokens.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= 2 and token not in _ENGLISH_STOPWORDS:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def chinese_sequences(text: str, *, minimum: int = 3) -> list[str]:
    """Return contiguous CJK sequences suitable for SQLite trigram MATCH."""

    sequences: list[str] = []
    current: list[str] = []
    for character in str(text):
        if _is_cjk(character):
            current.append(character)
        elif current:
            if len(current) >= minimum:
                sequences.append("".join(current))
            current = []
    if len(current) >= minimum:
        sequences.append("".join(current))
    return list(dict.fromkeys(sequences))


def fts_or_query(terms: Sequence[str]) -> str:
    """Build an FTS5 OR query from already-tokenized, safely quoted terms."""

    quoted = []
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        quoted.append('"' + normalized.replace('"', '""') + '"')
    return " OR ".join(dict.fromkeys(quoted))


def weighted_rrf(
    rankings: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float],
    rank_constant: float = 60.0,
    scale: float = 100.0,
) -> dict[str, float]:
    """Fuse independent ranked channels without comparing their raw scores."""

    scores: dict[str, float] = {}
    for channel, ranking in rankings.items():
        weight = float(weights.get(channel, 0.0))
        if weight <= 0:
            continue
        seen: set[str] = set()
        for rank, reference in enumerate(ranking, start=1):
            if reference in seen:
                continue
            seen.add(reference)
            scores[reference] = scores.get(reference, 0.0) + (
                scale * weight / (rank_constant + rank)
            )
    return scores


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )
