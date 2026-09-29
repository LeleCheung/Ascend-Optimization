"""Private detached worker entry point."""

from __future__ import annotations

import argparse

from kernelgen.cli.runner import execute_request
from kernelgen.cli.state import load_request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return execute_request(load_request(args.workspace), resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
