"""Reference gathering for the TestWriter agent.

Collects the pieces an LLM needs to write a *new-spec* FlagGems test:

  1. Public shape/dtype collections from ``tests/accuracy_utils.py``
  2. Timing shapes from ``benchmark/core_shapes.yaml`` (exact match + similar ops)
  3. Existing tests of *similar* operators as few-shot examples (tests/ + benchmark/)

Nothing here writes to FlagGems — it only reads, so it's safe to call from
preprocess.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

_ACCURACY_UTILS = "tests/accuracy_utils.py"
_CORE_SHAPES = "benchmark/core_shapes.yaml"


# ---------------------------------------------------------------------------
# accuracy_utils public collections
# ---------------------------------------------------------------------------

def extract_accuracy_constants(flaggems_dir: Path, limit: int = 40) -> str:
    """Extract public (UPPER_CASE) constants from tests/accuracy_utils.py.

    Returns a block like::

        <accuracy_utils_constants>
        POINTWISE_SHAPES = (() , (1,) , (1024, 1024), ...)
        FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
        ...
        </accuracy_utils_constants>
    """
    path = flaggems_dir / _ACCURACY_UTILS
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    constants = re.findall(
        r"^([A-Z][A-Z_]*)\s*=\s*(.+?)(?=\n[A-Z]|\n\n|\Z)",
        content, re.MULTILINE | re.DOTALL,
    )
    sections: List[str] = []
    for name, value in constants[:limit]:
        clean = value.strip().rstrip(",")
        if len(clean) < 400:
            sections.append(f"{name} = {clean}")
    if not sections:
        return ""
    return (
        "<accuracy_utils_constants>\n"
        "Public shape/dtype collections from tests/accuracy_utils.py — "
        "reuse these for correctness tests.\n\n"
        + "\n\n".join(sections) + "\n</accuracy_utils_constants>"
    )


# ---------------------------------------------------------------------------
# core_shapes.yaml timing shapes
# ---------------------------------------------------------------------------

def extract_core_shapes(flaggems_dir: Path, operator: str) -> str:
    """Pull timing shapes for ``operator`` (exact) + up to 5 similar ops."""
    path = flaggems_dir / _CORE_SHAPES
    if not path.exists():
        return ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""

    sections: List[str] = []
    base = operator.rstrip("_").removesuffix("_backward")
    if operator in data:
        entry = data[operator]
        sections.append(f"[timing] Exact '{operator}':\n  {entry}")

    similar = [
        (k, v) for k, v in data.items()
        if k != operator and k.startswith(base) and isinstance(v, dict) and "shapes" in v
    ]
    for k, v in similar[:5]:
        sections.append(
            f"[timing] Similar '{k}':\n  shapes: {v['shapes']}\n  desc: {v.get('shape_desc', '')}"
        )

    if not sections:
        return ""
    return (
        "<core_shapes_reference>\n"
        "Timing shapes from benchmark/core_shapes.yaml — use these for benchmark "
        "cases (performance-relevant shapes).\n\n"
        + "\n\n".join(sections) + "\n</core_shapes_reference>"
    )


# ---------------------------------------------------------------------------
# similar-operator test files (few-shot)
# ---------------------------------------------------------------------------

def _list_op_test_files(flaggems_dir: Path) -> List[str]:
    tests = sorted(p.name for p in (flaggems_dir / "tests").glob("test_*.py"))
    bench = sorted(p.name for p in (flaggems_dir / "benchmark").glob("test_*.py"))
    return tests + bench


def find_similar_ops(flaggems_dir: Path, operator: str, limit: int = 3) -> List[str]:
    """Return names of existing test files similar to ``operator``.

    Similarity: same base prefix (e.g. ``clamp_max`` → ``clamp_min``, ``clamp``),
    or matching a leading word (``layer_norm`` → ``rms_norm``, ``group_norm``).
    """
    all_files = _list_op_test_files(flaggems_dir)
    base = operator.rstrip("_").removesuffix("_backward").removesuffix("_out")
    seen = set()
    ranked: List[Tuple[int, str]] = []

    def score(fname: str) -> int:
        stem = fname.replace("test_", "").removesuffix(".py")
        if stem == operator:
            return 100
        if stem == base:
            return 90
        if base in stem or stem in base:
            return 60
        # shared leading word: layer_norm vs group_norm vs rms_norm
        for word in base.split("_"):
            if word and len(word) >= 3 and word in stem.split("_"):
                return 40
        return 0

    for fname in all_files:
        s = score(fname)
        if s > 0 and fname not in seen:
            seen.add(fname)
            ranked.append((s, fname))
    ranked.sort(reverse=True)
    # Exclude the operator's own test files (they are the target, not an example)
    return [
        fname for _, fname in ranked[:limit]
        if fname != f"test_{operator}.py"
    ][:limit]


def extract_similar_tests(flaggems_dir: Path, operator: str, limit: int = 2) -> str:
    """Few-shot: read similar operators' tests + benchmarks as examples.

    Returns a block like::

        <similar_operator_examples>
        == tests/test_clamp.py ==
        <code>
        ...
        </code>
        == benchmark/test_clamp.py ==
        <code>
        ...
        </code>
        </similar_operator_examples>
    """
    names = find_similar_ops(flaggems_dir, operator, limit=limit)
    blocks: List[str] = []
    for fname in names:
        for sub in ("tests", "benchmark"):
            p = flaggems_dir / sub / fname
            if p.exists():
                code = p.read_text(encoding="utf-8")
                blocks.append(f"== {sub}/{fname} ==\n```python\n{code}\n```")
    if not blocks:
        return ""
    return (
        "<similar_operator_examples>\n"
        "Existing FlagGems tests for similar operators — follow the same "
        "structure and conventions (these are new-spec style).\n\n"
        + "\n\n".join(blocks) + "\n</similar_operator_examples>"
    )


# ---------------------------------------------------------------------------
# aggregated
# ---------------------------------------------------------------------------

def build_reference_context(
    flaggems_dir: Path, operator: str, *, include_similar: bool = True
) -> str:
    """Aggregate all reference blocks for prompt injection."""
    parts = [
        extract_accuracy_constants(flaggems_dir),
        extract_core_shapes(flaggems_dir, operator),
    ]
    if include_similar:
        parts.append(extract_similar_tests(flaggems_dir, operator))
    return "\n\n".join(p for p in parts if p)
