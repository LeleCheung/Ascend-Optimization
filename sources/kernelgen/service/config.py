"""Service configuration resolved from the environment.

Defaults keep the server bound to loopback with no authentication so local
development just works. Setting ``KG_SERVICE_TOKEN`` turns on bearer-token auth;
``KG_SERVICE_HOST``/``KG_SERVICE_PORT`` override the bind address.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8378

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class ServiceConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str | None = None

    @property
    def requires_auth(self) -> bool:
        return bool(self.token)

    @property
    def is_loopback(self) -> bool:
        return self.host in _LOOPBACK

    @property
    def exposed_without_auth(self) -> bool:
        """True when the server listens off-loopback with no token set."""
        return not self.is_loopback and not self.requires_auth


def load_config() -> ServiceConfig:
    host = os.environ.get("KG_SERVICE_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    raw_port = os.environ.get("KG_SERVICE_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError as exc:
        raise ValueError(f"KG_SERVICE_PORT must be an integer, got {raw_port!r}") from exc
    token = os.environ.get("KG_SERVICE_TOKEN", "").strip() or None
    return ServiceConfig(host=host, port=port, token=token)
