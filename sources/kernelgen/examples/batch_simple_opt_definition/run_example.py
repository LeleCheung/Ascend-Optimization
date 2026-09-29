"""Batch example: kg run --mode simple_opt --batch-file <batch.yaml>."""

import sys


def main(argv=None):
    from kernelgen.cli.main import main as kg_main

    return kg_main(["run", "--mode", "simple_opt", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
