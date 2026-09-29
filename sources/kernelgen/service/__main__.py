"""Entrypoint: ``python -m kernelgen.service`` starts the HTTP service."""

from __future__ import annotations

import sys

import uvicorn

from kernelgen.service.app import create_app
from kernelgen.service.config import load_config


def main() -> int:
    config = load_config()
    if config.exposed_without_auth:
        print(
            f"WARNING: binding to {config.host} with no KG_SERVICE_TOKEN set. "
            "This exposes an unauthenticated run-control API to the network. "
            "Set KG_SERVICE_TOKEN or bind to 127.0.0.1.",
            file=sys.stderr,
        )
    app = create_app(config)
    print(f"KernelGen service on http://{config.host}:{config.port}", file=sys.stderr)
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
