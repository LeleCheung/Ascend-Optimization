"""HTTP routes for content-addressed v6.2 native operator uploads."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..operator_bundles import (
    OPERATOR_BUNDLE_MEDIA_TYPE,
    OperatorBundleError,
    OperatorBundleInfo,
    OperatorBundleStore,
    OperatorBundleTooLarge,
    normalize_operator_bundle_sha256,
)


def create_operator_bundle_router(store: OperatorBundleStore):
    try:
        from fastapi import APIRouter, HTTPException, Request, Response
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'server' extra to run the HTTP service") from exc

    router = APIRouter(prefix="/operator-bundles", tags=["operator-bundles"])

    def normalized_digest(value: str) -> str:
        try:
            return normalize_operator_bundle_sha256(value)
        except OperatorBundleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def existing_bundle(value: str) -> OperatorBundleInfo:
        digest = normalized_digest(value)
        try:
            info = store.get(digest)
        except OperatorBundleError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if info is None:
            raise HTTPException(status_code=404, detail="operator bundle not found")
        return info

    def response_headers(info: OperatorBundleInfo) -> dict[str, str]:
        return {
            "ETag": f'"{info.bundle_id}"',
            "X-KernelGen-Bundle-Id": info.bundle_id,
            "X-KernelGen-Definition": info.definition,
        }

    @router.head("/{bundle_sha256}")
    async def head_operator_bundle(bundle_sha256: str):
        info = existing_bundle(bundle_sha256)
        return Response(status_code=200, headers=response_headers(info))

    @router.get("/{bundle_sha256}", response_model=OperatorBundleInfo)
    async def get_operator_bundle(bundle_sha256: str) -> OperatorBundleInfo:
        return existing_bundle(bundle_sha256)

    async def put_operator_bundle(bundle_sha256: str, request):
        digest = normalized_digest(bundle_sha256)
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
        if media_type not in {OPERATOR_BUNDLE_MEDIA_TYPE, "application/x-tar"}:
            raise HTTPException(
                status_code=415,
                detail=f"operator bundle Content-Type must be {OPERATOR_BUNDLE_MEDIA_TYPE}",
            )
        declared_size = request.headers.get("content-length")
        if declared_size is not None:
            try:
                parsed_size = int(declared_size)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
            if parsed_size < 0:
                raise HTTPException(status_code=400, detail="invalid Content-Length")
            if parsed_size > store.max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"operator bundle exceeds {store.max_bytes} bytes",
                )

        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            suffix=".tar",
            dir=store.root,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        received = 0
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                received += len(chunk)
                if received > store.max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"operator bundle exceeds {store.max_bytes} bytes",
                    )
                temporary.write(chunk)
            temporary.close()
            try:
                info, created = await asyncio.to_thread(
                    store.install,
                    temporary_path,
                    digest,
                )
            except OperatorBundleTooLarge as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except OperatorBundleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return JSONResponse(
                status_code=201 if created else 200,
                content=info.model_dump(mode="json"),
                headers=response_headers(info),
            )
        finally:
            if not temporary.closed:
                temporary.close()
            temporary_path.unlink(missing_ok=True)

    put_operator_bundle.__annotations__["request"] = Request
    router.add_api_route(
        "/{bundle_sha256}",
        put_operator_bundle,
        methods=["PUT"],
        response_model=OperatorBundleInfo,
    )
    return router


__all__ = ["create_operator_bundle_router"]
