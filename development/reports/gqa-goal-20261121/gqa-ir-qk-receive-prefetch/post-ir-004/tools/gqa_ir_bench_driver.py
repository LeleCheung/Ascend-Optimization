"""Install an exact BCMLIR binding, then run the original GQA benchmark script."""
import argparse
from pathlib import Path
import runpy
import sys

from gqa_bcmlir_runtime import install


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    a, forwarded = p.parse_known_args()
    m = install(a.manifest)
    assert Path(forwarded[forwarded.index('--candidate') + 1]).resolve() == Path(m['source']).resolve()
    assert Path(forwarded[forwarded.index('--contract') + 1]).resolve() == Path(m['contract']).resolve()
    sys.argv = [str(a.harness), *forwarded]
    runpy.run_path(str(a.harness), run_name='__main__')


if __name__ == '__main__':
    main()
