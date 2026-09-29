"""KernelGen local command-line interface."""


def main(argv=None):
    from kernelgen.cli.main import main as cli_main

    return cli_main(argv)


__all__ = ["main"]
