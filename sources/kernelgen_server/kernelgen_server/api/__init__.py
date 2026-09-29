"""HTTP application assembly."""

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Load accelerator dependencies only when an app is constructed."""

    from .app import create_app as create

    return create(*args, **kwargs)

__all__ = ["create_app"]
