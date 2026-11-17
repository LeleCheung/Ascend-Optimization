"""Offline frozen trace import, source binding and presentation."""
import argparse
import json
from pathlib import Path

from .importer import build_session
from .index import build_index, save, verify
from .render import export_png, render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--presentation", type=Path, help="Build a trace index from selected frozen sessions")
    parser.add_argument("--index", type=Path, help="Verify an existing trace index")
    parser.add_argument("--render", type=Path, help="Directory for HTML and SVG exports")
    parser.add_argument("--png", action="store_true", help="Also export PNG using optional Playwright and Edge")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    if args.png and not args.render:
        parser.error("--png requires --render")
    root = args.root.resolve()
    if args.presentation:
        if not args.output:
            parser.error("--presentation requires --output")
        build_index(args.presentation.resolve(), args.output.resolve(), root)
        index_path = args.output.resolve()
    elif args.index:
        index_path = args.index.resolve()
    elif args.run and args.session and args.output:
        result=build_session(args.run.resolve(), root, args.session)
        save(args.output, result)
        print(json.dumps(dict(calls=len(result["calls"]),transitions=len(result["transitions"]),
                             snapshots=len(result["snapshots"]),gaps=result["gaps"]),ensure_ascii=False))
        return
    else:
        parser.error("provide --presentation/--index, or --run --session --output")
    result = verify(index_path, root)
    save(index_path.parent / "verification.json", result)
    if args.render:
        render(index_path, args.render.resolve(), root)
        if args.png:
            export_png(index_path, args.render.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__=="__main__":
    main()
