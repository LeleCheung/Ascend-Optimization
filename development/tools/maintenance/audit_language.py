"""Check English engineering text while preserving declared historical identifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
EXTENSIONS = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".ps1",
              ".cpp", ".cc", ".c", ".h", ".hpp", ".js", ".ts", ".html", ".css"}


def audit(repository: Path) -> dict:
    repository = repository.resolve()
    policy = json.loads((repository / "development/manifests/english-language-scope.json").read_bytes())
    excluded = [repository / name for name in policy["excluded_roots"]]
    paths = {repository / name for name in policy["maintained_files"]}
    for name in policy["maintained_roots"]:
        for path in (repository / name).rglob("*"):
            if (path.is_file() and path.suffix.lower() in EXTENSIONS
                    and "__pycache__" not in path.parts
                    and not any(path.is_relative_to(root) for root in excluded)
                    and not any(part.endswith(".egg-info") for part in path.parts)):
                paths.add(path)

    knowledge_path = repository / "hgkda/src/hgkda/data/knowledge/mechanisms.json"
    records = json.loads(knowledge_path.read_bytes())["records"]
    actual = {row["id"]: row for row in records}
    allowed_lines = set()
    findings = []
    for binding in policy["provenance_identifiers"]:
        row = actual.get(binding["id"], {})
        if any(row.get(key) != value for key, value in binding.items()):
            findings.append({"path": knowledge_path.relative_to(repository).as_posix(),
                             "reason": "Historical provenance binding changed", "id": binding["id"]})
        allowed_lines.add('"source": ' + json.dumps(binding["source"], ensure_ascii=False) + ',')

    for path in sorted(paths):
        if not path.is_file():
            findings.append({"path": path.relative_to(repository).as_posix(), "reason": "Maintained file missing"})
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if CJK.search(line) and not (path == knowledge_path and line.strip() in allowed_lines):
                findings.append({"path": path.relative_to(repository).as_posix(), "line": number,
                                 "reason": "Chinese text in maintained engineering material"})
    return {"status": "passed" if not findings else "failed", "checked_files": len(paths),
            "retained_provenance_identifiers": len(allowed_lines),
            "immutable_record_roots": policy["immutable_record_roots"], "findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.repository)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
