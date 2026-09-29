"""Unit tests for authoritative best-score selection.

Pure Python, no torch/GPU:

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_selection.py -v
    # or:
    python tests/test_selection.py
"""

from kernelgen.data.selection import pick_best


# --- pick_best ------------------------------------------------------------

def test_pick_best_argmax():
    cands = [("a", 1.1), ("b", 1.3), ("c", 0.9)]
    assert pick_best(cands, key=lambda t: t[1]) == ("b", 1.3)


def test_pick_best_empty_returns_default():
    assert pick_best([], key=lambda x: x) is None
    assert pick_best([], key=lambda x: x, default="fallback") == "fallback"


def test_pick_best_tie_first_seen():
    # two maxima -> the earliest wins (Python max semantics)
    cands = [("first", 1.3), ("second", 1.3), ("low", 1.0)]
    assert pick_best(cands, key=lambda t: t[1])[0] == "first"


def test_pick_best_opaque_payload():
    obj_a, obj_b = object(), object()
    cands = [{"geo": 1.0, "obj": obj_a}, {"geo": 2.0, "obj": obj_b}]
    assert pick_best(cands, key=lambda d: d["geo"])["obj"] is obj_b
