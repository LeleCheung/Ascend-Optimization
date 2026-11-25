"""Resolve recorded repository relocations without rewriting original evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def relocation_records(repository: Path) -> list[dict]:
    engineering = json.loads((repository / "development/manifests/engineering-relocations-20261124.json").read_bytes())
    previous = json.loads((repository / "development/data/relocation-20261123.json").read_bytes())
    return (engineering["files"] +
            [dict(old=row["previous"], new=row["current"], sha256=row["sha256"]) for row in previous])


def resolve(repository: Path, name: str) -> dict:
    repository = repository.resolve()
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("Use a repository-relative path without traversal")
    current = path.as_posix()
    mapping = {row["old"]: row for row in relocation_records(repository)}
    chain, seen, expected = [], set(), None
    while current in mapping:
        if current in seen:
            raise ValueError("Relocation cycle: " + current)
        seen.add(current)
        row = mapping[current]
        if expected is not None and expected != row["sha256"]:
            raise ValueError("Relocation chain changed artifact identity")
        expected = row["sha256"]
        chain.append(current)
        current = row["new"]
    target = (repository / current).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise ValueError("Resolved artifact is missing or outside the repository: " + current)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Relocated artifact bytes changed: " + current)
    return {"requested": path.as_posix(), "path": current, "sha256": digest,
            "previous_paths": chain}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    try:
        result = resolve(args.repository, args.path)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
