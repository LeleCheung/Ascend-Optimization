"""FastAPI application exposing the shared run-control contract over HTTP.

Every endpoint is a thin wrapper over :mod:`kernelgen.cli.api`; there is no
business logic here beyond request parsing, auth, and error mapping. A run
submitted over HTTP is byte-for-byte equivalent to one from ``kg run``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kernelgen.cli import api
from kernelgen.cli.main import _default_workspace
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.service.config import ServiceConfig, load_config
from kernelgen.service.models import (
    CancelRequest,
    KgsInstanceCreateRequest,
    SubmitRunRequest,
    SubmitRunResponse,
)

_STATIC_DIR = Path(__file__).parent / "static"


def _make_auth_dependency(config: ServiceConfig):
    """Return a dependency that enforces the bearer token when one is set."""

    async def verify_token(
        authorization: str | None = Header(default=None),
        x_kg_token: str | None = Header(default=None),
    ) -> None:
        if not config.requires_auth:
            return
        presented = x_kg_token
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[len("bearer ") :].strip()
        if presented != config.token:
            raise HTTPException(status_code=401, detail="invalid or missing token")

    return verify_token


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="KernelGen Service", version="1.0")
    app.state.config = config
    auth = Depends(_make_auth_dependency(config))

    @app.post("/api/runs", response_model=SubmitRunResponse, dependencies=[auth])
    def submit(payload: SubmitRunRequest) -> SubmitRunResponse:
        try:
            values = resolve_run_options(payload.options)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace = (
            Path(payload.workspace).expanduser().resolve()
            if payload.workspace
            else _default_workspace(payload.definition)
        )
        try:
            request, record = api.submit_run(
                values, definition=payload.definition, workspace=workspace, foreground=False
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SubmitRunResponse(
            workspace=str(request.workspace),
            run_id=request.run_id,
            pid=record.pid,
            status="SUBMITTED",
        )

    @app.get("/api/runs", dependencies=[auth])
    def list_runs() -> dict:
        return {"schema_version": "2.0", "runs": api.list_run_statuses()}

    @app.get("/api/status", dependencies=[auth])
    def get_status(workspace: str = Query(min_length=1)) -> dict:
        try:
            return api.status(workspace)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/history", dependencies=[auth])
    def get_history(workspace: str = Query(min_length=1)) -> dict:
        try:
            return api.history(workspace)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/activity", dependencies=[auth])
    def get_activity(
        workspace: str = Query(min_length=1),
        after_sequence: int | None = Query(default=None, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict:
        try:
            return api.run_activity(
                workspace,
                after_sequence=after_sequence,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/cancel", dependencies=[auth])
    def cancel(payload: CancelRequest) -> dict:
        try:
            return api.cancel_run(payload.workspace, payload.reason, source="http:cancel")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/config", dependencies=[auth])
    def get_config() -> dict:
        from kernelgen.cli.server import _kg_release

        return {"requires_auth": config.requires_auth, "version": _kg_release()}

    @app.get("/api/worker-pools", dependencies=[auth])
    def worker_pools(eval_server: str = Query(default="")) -> dict:
        try:
            pools = (
                [api.worker_pool_status(eval_server)]
                if eval_server.strip()
                else api.list_worker_pools()
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"schema_version": "1.0", "pools": pools}

    @app.get("/api/device-instances", dependencies=[auth])
    def device_instances() -> dict:
        from kernelgen.service.fleet_daemon import load_public_registry

        return load_public_registry()

    @app.get("/api/kgs-versions", dependencies=[auth])
    def kgs_versions() -> dict:
        from kernelgen.service.kgs_manager import list_versions

        try:
            return list_versions()
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail="KGS versions are temporarily unavailable",
            ) from exc

    @app.get("/api/kgs-deployments", dependencies=[auth])
    def kgs_deployments() -> dict:
        from kernelgen.service.kgs_manager import list_deployments

        return list_deployments()

    @app.post("/api/kgs-deployments/{deployment_id}/stop", dependencies=[auth])
    def stop_kgs_deployment(deployment_id: str) -> dict:
        from kernelgen.service.kgs_manager import stop_deployment

        try:
            return stop_deployment(deployment_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="managed KGS deployment not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail="KGS deployment is not ready to stop",
            ) from exc
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail="KGS stop is temporarily unavailable",
            ) from exc

    @app.post("/api/kgs-instances", status_code=202, dependencies=[auth])
    def create_kgs_instance(
        payload: KgsInstanceCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> dict:
        from kernelgen.service.kgs_manager import deploy, reserve_deployment

        try:
            record, task = reserve_deployment(
                payload.machine,
                payload.version,
                install_flaggems=payload.install_flaggems,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="machine not found") from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="invalid machine or KGS version",
            ) from exc
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail="KGS deployment is temporarily unavailable",
            ) from exc
        background_tasks.add_task(deploy, task)
        return record

    @app.get("/api/devices", dependencies=[auth])
    def devices() -> dict:
        from kernelgen.service.hints import list_devices
        return {"schema_version": "1.0", "devices": list_devices()}

    @app.get("/api/definitions", dependencies=[auth])
    def definitions(catalog: str = Query(default="")) -> dict:
        from kernelgen.service.hints import list_definitions
        return {"schema_version": "1.0", **list_definitions(catalog)}

    @app.get("/", response_model=None)
    def index() -> FileResponse | JSONResponse:
        page = _STATIC_DIR / "index.html"
        if not page.is_file():
            return JSONResponse({"service": "kernelgen", "ui": "not installed"})
        return FileResponse(page)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app
