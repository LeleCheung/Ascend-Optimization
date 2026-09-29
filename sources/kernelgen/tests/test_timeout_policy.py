import pytest

from kernelgen.data.timeout_policy import TimeoutPolicy


def test_default_timeout_policy_derives_all_internal_budgets():
    policy = TimeoutPolicy()

    assert policy.eval_timeout_seconds == 1500
    assert policy.eval_transport_timeout_seconds == 1800
    assert policy.coder_idle_timeout_seconds == 2100
    assert policy.coder_hard_timeout_seconds == 3600


def test_timeout_policy_preserves_budget_order_for_custom_eval_limit():
    policy = TimeoutPolicy(900)

    assert policy.eval_transport_timeout_seconds == 1200
    assert policy.coder_idle_timeout_seconds == 1500
    assert policy.coder_hard_timeout_seconds == 3000
    assert (
        policy.eval_timeout_seconds
        < policy.eval_transport_timeout_seconds
        < policy.coder_idle_timeout_seconds
        < policy.coder_hard_timeout_seconds
    )


def test_timeout_policy_rejects_non_positive_eval_limit():
    with pytest.raises(ValueError, match="must be positive"):
        TimeoutPolicy(0)
