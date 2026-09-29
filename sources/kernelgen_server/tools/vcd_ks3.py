#!/usr/bin/env python3
"""Small, dependency-free KS3 and safetensors helpers for VCD catalog tools."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SafetensorsHeader:
    header: dict
    header_length: int
    object_size: int


class KS3Client:
    """Minimal read-only client for Kingsoft Cloud KS3's KSS V2 API."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        ak: str,
        sk: str,
        prefix: str = "",
        timeout: float = 90.0,
        retries: int = 4,
    ) -> None:
        if not all((endpoint, bucket, ak, sk)):
            raise ValueError("endpoint, bucket, ak and sk must be non-empty")
        self.endpoint = endpoint.strip().rstrip("/")
        self.bucket = bucket.strip()
        self.ak = ak
        self.sk = sk
        self.prefix = prefix.strip("/")
        self.timeout = timeout
        self.retries = retries

    @staticmethod
    def _date() -> str:
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        if not key:
            raise ValueError("KS3 object key must be non-empty")
        return f"{self.prefix}/{key}" if self.prefix else key

    def _request(self, key: str, *, byte_range: tuple[int, int] | None = None):
        full_key = self._full_key(key)
        resource = f"/{self.bucket}/{full_key}"
        date = self._date()
        canonical = f"GET\n\n\n{date}\n{resource}"
        signature = base64.b64encode(
            hmac.new(self.sk.encode(), canonical.encode(), hashlib.sha1).digest()
        ).decode()
        headers = {"Date": date, "Authorization": f"KSS {self.ak}:{signature}"}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        url = (
            f"https://{self.bucket}.{self.endpoint}/"
            f"{quote(full_key, safe='/-_.~')}"
        )
        request = Request(url, headers=headers, method="GET")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return urlopen(request, timeout=self.timeout)
            except HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    body = exc.read(500).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"KS3 GET failed [{exc.code}] for {key}: {body}"
                    ) from exc
            except URLError as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"KS3 GET failed for {key}: {last_error}") from last_error

    def download(self, key: str) -> bytes:
        with self._request(key) as response:
            return response.read()

    def header(self, key: str) -> SafetensorsHeader:
        end = 65535
        with self._request(key, byte_range=(0, end)) as response:
            data = response.read()
            content_range = response.headers.get("Content-Range")
            object_size = _object_size(content_range, response.headers.get("Content-Length"))
        if len(data) < 8:
            raise ValueError(f"invalid safetensors object {key}: missing header length")
        header_length = struct.unpack("<Q", data[:8])[0]
        if header_length <= 0:
            raise ValueError(f"invalid safetensors object {key}: empty header")
        if len(data) < 8 + header_length:
            with self._request(key, byte_range=(0, 7 + header_length)) as response:
                data = response.read()
                content_range = response.headers.get("Content-Range")
                object_size = _object_size(
                    content_range, response.headers.get("Content-Length")
                )
        try:
            header = json.loads(data[8 : 8 + header_length])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors JSON header in {key}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors header object in {key}")
        return SafetensorsHeader(header, header_length, object_size)


def _object_size(content_range: str | None, content_length: str | None) -> int:
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    if content_length:
        return int(content_length)
    raise ValueError("KS3 response did not include an object size")


def split_safetensors(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 8:
        raise ValueError("invalid safetensors payload: missing header length")
    header_length = struct.unpack("<Q", data[:8])[0]
    if header_length <= 0 or 8 + header_length > len(data):
        raise ValueError("invalid safetensors payload: truncated header")
    header = json.loads(data[8 : 8 + header_length])
    return header, data[8 + header_length :]


def rename_safetensors(
    data: bytes, destination_to_source: dict[str, str | None]
) -> bytes:
    """Rename Tensor keys and materialize fixed-ABI empty output slots."""
    header, tensor_data = split_safetensors(data)
    missing = {
        source for source in destination_to_source.values() if source is not None
    }.difference(header)
    if missing:
        raise ValueError(f"source safetensors is missing keys: {sorted(missing)}")
    tensor_entries = [
        value
        for key, value in header.items()
        if key not in {"__metadata__", "__empty__"} and isinstance(value, dict)
    ]
    data_end = max(
        (int(entry["data_offsets"][1]) for entry in tensor_entries), default=0
    )
    converted = {}
    for destination, source in destination_to_source.items():
        if source is None:
            converted[destination] = {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [data_end, data_end],
            }
        else:
            converted[destination] = header[source]
    raw_header = json.dumps(
        converted, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    return struct.pack("<Q", len(raw_header)) + raw_header + tensor_data


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(data)
    temporary.replace(path)
