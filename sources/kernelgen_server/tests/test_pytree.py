import pytest

torch = pytest.importorskip("torch")

from kernelgen_server.evaluation.pytree import clone, structure


def test_nested_structure_and_repeated_alias():
    tensor = torch.arange(4)
    tree = {"out": [tensor, (tensor, None)]}
    copied = clone(tree)
    assert structure(copied) == structure(tree)
    assert copied["out"][0] is copied["out"][1][0]
    assert copied["out"][0] is not tensor


def test_clone_preserves_lazy_negative_bit():
    source = torch._neg_view(torch.tensor([3.0]))
    copied = clone(source)
    assert source.is_neg()
    assert copied.is_neg()
    assert copied.item() == -3.0
