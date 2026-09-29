#!/usr/bin/env python3
"""Download VCD inputs and Golden outputs for vLLM 0.28 Triton."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from vcd_ks3 import KS3Client, atomic_write, rename_safetensors


# This repository is private.  Environment variables still take precedence so
# credentials can be rotated without editing the script.
DEFAULT_KS3_AK = "AKLTYHBEOuZWQVSwrrPUjhta"
DEFAULT_KS3_SK = "OJ6eRPmd3xONNzJZzb0qBJF4jvjPACdJOYyjEfFb"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _local_path(canonical: str, dump_root: str, output_root: Path) -> Path:
    path = PurePosixPath(canonical)
    base = PurePosixPath(dump_root)
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path {canonical!r} is outside manifest dump_root {dump_root!r}") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe manifest path: {canonical!r}")
    return output_root.joinpath(*relative.parts)


def _local_tensor_keys(path: Path) -> set[str]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            return set()
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return set()
    return set(header).difference({"__metadata__", "__empty__"})


def _download_case(
    client: KS3Client,
    item: dict[str, Any],
    *,
    manifest_dump_root: str,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, int, int]:
    operator = str(item["operator"])
    case_index = int(item["case_index"])
    input_spec = item["input"]
    output_spec = item["output"]
    input_path = _local_path(input_spec["path"], manifest_dump_root, output_root)
    output_path = _local_path(output_spec["path"], manifest_dump_root, output_root)
    if dry_run:
        return f"{operator}:{case_index}", 0, 0

    downloaded = 0
    reused = 0
    input_size = int(input_spec["source_size"])
    if not overwrite and input_path.is_file() and input_path.stat().st_size == input_size:
        reused += 1
    else:
        payload = client.download(str(input_spec["key"]))
        if len(payload) != input_size:
            raise ValueError(
                f"input size mismatch for {operator}:{case_index}: "
                f"expected {input_size}, got {len(payload)}"
            )
        atomic_write(input_path, payload)
        downloaded += 1

    key_map = output_spec["destination_to_source"]
    expected_keys = set(key_map)
    if (
        not overwrite
        and output_path.is_file()
        and _local_tensor_keys(output_path) == expected_keys
    ):
        reused += 1
    else:
        payload = client.download(str(output_spec["key"]))
        source_size = int(output_spec["source_size"])
        if len(payload) != source_size:
            raise ValueError(
                f"output size mismatch for {operator}:{case_index}: "
                f"expected {source_size}, got {len(payload)}"
            )
        atomic_write(output_path, rename_safetensors(payload, key_map))
        downloaded += 1
    return f"{operator}:{case_index}", downloaded, reused


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    default_catalog = repository / "data" / "kernelgenbench-vcd-vllm28-triton"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=default_catalog / "ks3_objects.json"
    )
    parser.add_argument("--output-root", type=Path, default=Path("/data/dumps"))
    parser.add_argument(
        "--operator", action="append", default=[], help="download only this operator"
    )
    parser.add_argument("--limit", type=int, help="download at most this many cases")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    manifest = _read_json(args.manifest)
    ks3 = manifest.get("ks3")
    objects = manifest.get("objects")
    dump_root = manifest.get("dump_root")
    if not isinstance(ks3, dict) or not isinstance(objects, list) or not isinstance(dump_root, str):
        raise ValueError(f"invalid VCD download manifest: {args.manifest}")
    selected = [
        item
        for item in objects
        if isinstance(item, dict)
        and (not args.operator or item.get("operator") in set(args.operator))
    ]
    if args.operator:
        found = {str(item["operator"]) for item in selected}
        missing = set(args.operator).difference(found)
        if missing:
            raise ValueError(f"unknown operators: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: args.limit]

    ak = os.environ.get("VCD_KS3_AK", DEFAULT_KS3_AK)
    sk = os.environ.get("VCD_KS3_SK", DEFAULT_KS3_SK)
    if not ak or not sk or ak.startswith("__VCD_") or sk.startswith("__VCD_"):
        raise RuntimeError("KS3 credentials are not configured in the script or environment")
    client = KS3Client(
        endpoint=str(ks3["endpoint"]),
        bucket=str(ks3["bucket"]),
        prefix=str(ks3.get("prefix", "")),
        ak=ak,
        sk=sk,
    )
    print(
        f"downloading {len(selected)} cases ({len(selected) * 2} files) "
        f"to {args.output_root}"
    )
    downloaded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _download_case,
                client,
                item,
                manifest_dump_root=dump_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            for item in selected
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            _, new_count, reused_count = future.result()
            downloaded += new_count
            reused += reused_count
            if completed % 50 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)} cases", flush=True)
    print(f"done: downloaded={downloaded}, reused={reused}, cases={len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
