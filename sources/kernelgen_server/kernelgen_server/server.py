"""Compatibility CLI and imports for the HTTP application."""

from typing import Any

from .api.cli import main


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper that defers accelerator imports."""

    from .api.app import create_app as create

    return create(*args, **kwargs)


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
