"""Tests for the HTTP service layer (kernelgen.service).

The service is a thin wrapper over kernelgen.cli.api, so these tests focus on
HTTP concerns: request parsing, auth, error mapping, and that endpoints reach
the shared contract. Real runs are seeded on disk (as in test_cli.py) or the
submit call is monkeypatched to avoid spawning worker processes.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from kernelgen.cli.batch import save_batch_request
from kernelgen.cli.models import (
    BatchChildRecord,
    BatchRequestRecord,
    ProcessState,
    RunProcessRecord,
    utc_now,
)
from kernelgen.cli.runner import new_request
from kernelgen.cli.state import (
    process_start_identity,
    register_workspace,
    runner_log_path,
    save_process,
    save_request,
    set_max_workers,
)
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.service.app import create_app
from kernelgen.service.config import ServiceConfig


def _client(config: ServiceConfig | None = None) -> TestClient:
    return TestClient(create_app(config or ServiceConfig()))


def _seed_run(tmp_path, monkeypatch, *, state=ProcessState.QUEUED):
    """Persist a run on disk so status/list/cancel have something to read."""
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://cuda-kgs:8000", 1)
    workspace = tmp_path / "run"
    request = new_request(
        mode="simple_opt",
        definition="demo",
        workspace=workspace,
        workflow_args=[],
        n_parallel=1,
        target_hardware="A100",
        eval_server="http://cuda-kgs:8000",
    )
    workspace.mkdir()
    save_request(request)
    register_workspace(workspace)
    identity = process_start_identity(os.getpid())
    assert identity is not None
    save_process(
        RunProcessRecord(
            run_id=request.run_id,
            pid=os.getpid(),
            process_start=identity,
            state=state,
            workspace=workspace,
            log_path=runner_log_path(workspace),
            worker_weight=1,
            submitted_at=utc_now(),
        )
    )
    return request, workspace


def test_config_endpoint_reports_no_auth_by_default():
    resp = _client().get("/api/config")
    assert resp.status_code == 200
    assert resp.json() == {"requires_auth": False, "version": "v6.5.0"}


def test_console_keeps_one_detail_panel_for_inline_run_expansion():
    resp = _client().get("/")

    assert resp.status_code == 200
    html = resp.text
    assert html.count('id="detail"') == 1
    assert '<div id="detailParking" hidden>' in html
    assert '<div class="detail" id="detail" hidden></div>' in html
    assert 'entry.className = "run-entry"' in html
    assert "function reconcileRunList(list, runs)" in html
    assert "function parkDetail()" in html
    assert "function mountDetail()" in html
    assert "if (ws === selectedWs)" in html
    assert "clearRunSelection();" in html
    assert '"SUCCEEDED", "GENERATED", "FAILED"' in html
    assert ".b-SUCCEEDED, .b-GENERATED" in html


def test_console_run_polling_updates_rows_without_remounting_detail():
    html = _client().get("/").text
    load_runs = html.split("async function loadRuns()", 1)[1].split(
        "async function selectRun", 1
    )[0]

    assert "reconcileRunList(list, runs);" in load_runs
    assert "parkDetail();" not in load_runs
    assert "mountDetail()" not in load_runs
    assert "list.innerHTML = runs.map" not in load_runs
    assert "selectedEntry.classList.add(\"expanded\")" in load_runs


def test_console_preserves_active_text_selection_during_polling():
    html = _client().get("/").text

    assert "function hasTextSelectionWithin(root)" in html
    assert "function detailSelectionActive(workspace)" in html
    assert "function setTextIfChanged(node, value)" in html
    assert "if (hasTextSelectionWithin(row)) return;" in html
    assert "if (!detailSelectionActive(selectedWs)) refreshDetail();" in html
    assert "if (detailSelectionActive(requestWs)) return;" in html
    assert 'row.addEventListener("click", () => {' in html
    assert "stateBadge.textContent = state;" not in html


def test_console_run_rows_show_best_instance_and_rounds():
    html = _client().get("/").text

    assert "function runProgressDetails(run)" in html
    assert 'root.scopes && root.scopes["stages/optimize"]' in html
    assert "item && item.progress" in html
    assert "function runInstanceLabel(run)" in html
    assert "item.eval_server === run.eval_server" in html
    assert 'class="metric best-metric"' in html
    assert 'class="metric best"' not in html
    assert 'class="metric-value metric-best"' in html
    assert 'class="metric-value metric-instance"' in html
    assert 'class="metric-value metric-rounds"' in html
    assert "`${Number(best).toFixed(3)}×`" in html
    assert "`${Number(rounds)} rounds`" in html


def test_console_preserves_activity_scroll_across_detail_refreshes():
    html = _client().get("/").text

    assert "let activityViewState" in html
    assert "function captureActivityScroll()" in html
    assert "function restoreActivityScroll()" in html
    assert "function createActivitySection()" in html
    assert "function syncActivitySection(section)" in html
    assert "anchorSequence" in html
    assert "saved.pinned" in html
    assert 'feed.addEventListener("scroll", captureActivityScroll' in html
    assert 'feed.removeEventListener("scroll", captureActivityScroll)' in html
    assert "feed.clientHeight === 0" in html
    assert 'const preservedFeed = $("activityFeed")' in html
    assert 'const preservedActivity = preservedFeed ? preservedFeed.closest' in html
    assert 'preservedActivity || createActivitySection()' in html
    assert "activitySlot.parentNode.replaceChild(activitySection, activitySlot)" in html
    assert "detail.hidden = false;\n  restoreActivityScroll();" in html
    assert "feed.appendChild(row);" in html
    assert "${renderActivity(s)}" not in html
    assert "restoreActivityScroll();" in html


def test_console_renders_speedup_curve_and_highlights_run_best():
    html = _client().get("/").text

    assert "function renderSpeedupChart(history, series)" in html
    assert 'class="speedup-line"' in html
    assert 'class="speedup-point${point.best ? " best-point" : ""}"' in html
    assert 'class="speedup-best-label"' in html
    assert "best ${best.yValue.toFixed(3)}x" in html
    assert "function isRunBest(history, series, round)" in html
    assert "series.map((item) => renderRoundSeries(history, item, showScope))" in html
    assert 'class="best-tag">★ RUN BEST' in html
    assert "No valid speedup measurements yet." in html


def test_submit_delegates_to_api_and_returns_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    captured = {}

    def fake_submit(values, *, definition, workspace, foreground):
        captured["values"] = values
        captured["definition"] = definition
        captured["foreground"] = foreground
        request = new_request(
            mode=values["mode"],
            definition=definition,
            workspace=workspace,
            workflow_args=[],
            n_parallel=values.get("n_parallel", 1),
            target_hardware=values["target_hardware"],
            eval_server=values["eval_server"],
        )
        record = RunProcessRecord(
            run_id=request.run_id,
            pid=os.getpid(),
            process_start=process_start_identity(os.getpid()),
            state=ProcessState.SUBMITTED,
            workspace=workspace,
            log_path=runner_log_path(workspace),
            worker_weight=1,
            submitted_at=utc_now(),
        )
        return request, record

    monkeypatch.setattr("kernelgen.service.app.api.submit_run", fake_submit)

    resp = _client().post(
        "/api/runs",
        json={
            "definition": "gcd",
            "options": {
                "mode": "simple_opt",
                "target_hardware": "A100",
                "eval_server": "http://cuda-kgs:8000",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "SUBMITTED"
    assert body["run_id"]
    assert body["workspace"]
    # Background submission, never foreground over HTTP.
    assert captured["foreground"] is False
    assert captured["definition"] == "gcd"
    assert captured["values"]["mode"] == "simple_opt"


def test_submit_rejects_bad_options_with_422():
    resp = _client().post(
        "/api/runs",
        json={"definition": "gcd", "options": {"mode": "not_a_mode"}},
    )
    assert resp.status_code == 422


def test_submit_uses_shared_default_mode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    captured = {}
    def submit(values, **kwargs):
        captured.update(values)
        return (SimpleNamespace(workspace=kwargs["workspace"], run_id="test-run"),
                SimpleNamespace(pid=123))
    monkeypatch.setattr("kernelgen.service.app.api.submit_run", submit)
    resp = _client().post("/api/runs", json={"definition": "gcd", "workspace": str(tmp_path / "run"), "options": {}})
    assert resp.status_code == 200
    assert captured["mode"] == "kernelgen"
    assert captured["n_parallel"] == captured["n_epoch"] == 1
    assert captured["max_round"] == 10


def test_status_reads_seeded_run(tmp_path, monkeypatch):
    request, workspace = _seed_run(tmp_path, monkeypatch)
    resp = _client().get("/api/status", params={"workspace": str(workspace)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == request.run_id
    assert body["definition"] == "demo"


def test_status_missing_workspace_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().get("/api/status", params={"workspace": str(tmp_path / "nope")})
    assert resp.status_code == 404


def test_list_runs_includes_seeded_run(tmp_path, monkeypatch):
    request, _ = _seed_run(tmp_path, monkeypatch)
    resp = _client().get("/api/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert any(item.get("run_id") == request.run_id for item in runs)


def test_worker_pools_empty_when_no_config(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().get("/api/worker-pools")
    assert resp.status_code == 200
    assert resp.json() == {"schema_version": "1.0", "pools": []}


def test_worker_pool_explicit_idle_uses_default_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().get(
        "/api/worker-pools", params={"eval_server": "http://localhost:18006"}
    )
    assert resp.status_code == 200
    pools = resp.json()["pools"]
    assert len(pools) == 1
    pool = pools[0]
    assert pool["worker_pool"] == "http://127.0.0.1:18006"
    assert pool["max_workers"] == 1
    assert pool["used_workers"] == 0
    assert pool["available_workers"] == 1
    assert pool["queued_count"] == 0
    assert pool["queued_workers"] == 0
    assert pool["queued_runs"] == []


def test_worker_pool_explicit_reports_waiting_run(tmp_path, monkeypatch):
    request, workspace = _seed_run(tmp_path, monkeypatch)
    resp = _client().get(
        "/api/worker-pools", params={"eval_server": "http://cuda-kgs:8000"}
    )
    assert resp.status_code == 200
    pool = resp.json()["pools"][0]
    assert pool["queued_count"] == 1
    assert pool["queued_workers"] == 1
    assert pool["queued_runs"][0]["run_id"] == request.run_id
    assert pool["queued_runs"][0]["workspace"] == str(workspace)


def test_worker_pools_report_configured_and_active(tmp_path, monkeypatch):
    # Seeding a run configures http://cuda-kgs:8000 and adds a live lease.
    _seed_run(tmp_path, monkeypatch)
    set_max_workers("http://h100-kgs:8000", 4)  # configured but idle
    resp = _client().get("/api/worker-pools")
    assert resp.status_code == 200
    pools = {p["worker_pool"]: p for p in resp.json()["pools"]}
    assert "http://cuda-kgs:8000" in pools
    assert "http://h100-kgs:8000" in pools
    assert pools["http://h100-kgs:8000"]["max_workers"] == 4
    assert pools["http://h100-kgs:8000"]["used_workers"] == 0


def test_worker_pool_invalid_eval_server_returns_422(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().get("/api/worker-pools", params={"eval_server": "not-a-url"})
    assert resp.status_code == 422


def test_device_instances_returns_sanitized_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    snapshot = {
        "schema_version": "1.0",
        "updated_at": "2026-09-10T00:00:00Z",
        "daemon_running": True,
        "devices": [
            {
                "name": "musa",
                "address": "10.1.2.3",
                "target_hardware": "S5000",
                "state": "ok",
            }
        ],
        "instances": [
            {
                "instance_id": "musa-18403",
                "device": "musa",
                "target_hardware": "S5000",
                "remote_port": 18403,
                "local_port": 18100,
                "eval_server": "http://127.0.0.1:18100",
                "status": "ok",
                "proxy_state": "running",
                "unavailable_reason": None,
                "selectable": True,
            }
        ],
    }
    monkeypatch.setattr(
        "kernelgen.service.fleet_daemon.load_public_registry", lambda: snapshot
    )
    resp = _client().get("/api/device-instances")
    assert resp.status_code == 200
    assert resp.json() == snapshot


def test_device_instances_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(
        "kernelgen.service.fleet_daemon.load_public_registry",
        lambda: {"schema_version": "1.0", "devices": [], "instances": []},
    )
    client = _client(ServiceConfig(token="s3cret"))
    assert client.get("/api/device-instances").status_code == 401
    assert (
        client.get(
            "/api/device-instances", headers={"X-KG-Token": "s3cret"}
        ).status_code
        == 200
    )


def test_kgs_versions_returns_version_labels_without_revisions(monkeypatch):
    monkeypatch.setattr(
        "kernelgen.service.kgs_manager.list_versions",
        lambda: {
            "schema_version": "1.0",
            "versions": [
                {"name": "v6.3.4", "recommended": True},
                {"name": "development", "recommended": False},
            ],
        },
    )

    resp = _client().get("/api/kgs-versions")

    assert resp.status_code == 200
    assert resp.json()["versions"][0] == {
        "name": "v6.3.4",
        "recommended": True,
    }
    assert "commit" not in resp.text.lower()
    assert "revision" not in resp.text.lower()


def test_kgs_version_failure_is_sanitized(monkeypatch):
    def fail():
        raise RuntimeError("private repository failure")

    monkeypatch.setattr("kernelgen.service.kgs_manager.list_versions", fail)

    resp = _client().get("/api/kgs-versions")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "KGS versions are temporarily unavailable"
    assert "private repository failure" not in resp.text


def test_create_kgs_instance_queues_private_task_and_returns_safe_record(monkeypatch):
    public = {
        "deployment_id": "fleet-ascend-18310",
        "instance_id": "ascend-18310",
        "machine": "ascend",
        "version": "development",
        "status": "starting",
        "error_category": None,
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
    }
    private_task = {
        "record": {"instance_name": "fleet-ascend-18310"},
        "resolved": {"commit": "a" * 40},
        "ssh_command": "private command",
    }
    queued = []
    reserved = []

    def reserve(machine, version, *, install_flaggems):
        reserved.append((machine, version, install_flaggems))
        return public, private_task

    monkeypatch.setattr(
        "kernelgen.service.kgs_manager.reserve_deployment",
        reserve,
    )
    monkeypatch.setattr(
        "kernelgen.service.kgs_manager.deploy", lambda task: queued.append(task)
    )

    resp = _client().post(
        "/api/kgs-instances",
        json={"machine": "ascend", "version": "development"},
    )

    assert resp.status_code == 202
    assert resp.json() == public
    assert reserved == [("ascend", "development", True)]
    assert queued == [private_task]
    assert "commit" not in resp.text.lower()
    assert "ssh" not in resp.text.lower()
    assert "remote_port" not in resp.text
    assert "local_port" not in resp.text


def test_create_kgs_instance_passes_explicit_native_only_choice(monkeypatch):
    captured = []
    public = {
        "deployment_id": "fleet-ascend-18310",
        "instance_id": "ascend-18310",
        "machine": "ascend",
        "version": "development",
        "status": "starting",
        "error_category": None,
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
    }
    task = {"record": {"instance_name": "fleet-ascend-18310"}}

    def reserve(machine, version, *, install_flaggems):
        captured.append(install_flaggems)
        return public, task

    monkeypatch.setattr("kernelgen.service.kgs_manager.reserve_deployment", reserve)
    monkeypatch.setattr("kernelgen.service.kgs_manager.deploy", lambda _task: None)

    resp = _client().post(
        "/api/kgs-instances",
        json={
            "machine": "ascend",
            "version": "development",
            "install_flaggems": False,
        },
    )

    assert resp.status_code == 202
    assert captured == [False]
    assert "install_flaggems" not in resp.text


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (KeyError("missing"), 404, "machine not found"),
        (ValueError("bad branch"), 422, "invalid machine or KGS version"),
        (RuntimeError("private failure"), 503, "KGS deployment is temporarily unavailable"),
    ],
)
def test_create_kgs_instance_maps_safe_errors(monkeypatch, error, status, detail):
    def fail(_machine, _version, *, install_flaggems):
        raise error

    monkeypatch.setattr("kernelgen.service.kgs_manager.reserve_deployment", fail)

    resp = _client().post(
        "/api/kgs-instances",
        json={"machine": "ascend", "version": "development"},
    )

    assert resp.status_code == status
    assert resp.json()["detail"] == detail
    assert str(error) not in resp.text


def test_kgs_deployments_returns_only_public_state(monkeypatch):
    snapshot = {
        "schema_version": "1.0",
        "deployments": [
            {
                "deployment_id": "fleet-ascend-18310",
                "instance_id": "ascend-18310",
                "machine": "ascend",
                "version": "development",
                "status": "waiting_for_fleet",
                "error_category": None,
                "created_at": "2026-09-16T00:00:00Z",
                "updated_at": "2026-09-16T00:00:01Z",
            }
        ],
    }
    monkeypatch.setattr(
        "kernelgen.service.kgs_manager.list_deployments", lambda: snapshot
    )

    resp = _client().get("/api/kgs-deployments")

    assert resp.status_code == 200
    assert resp.json() == snapshot
    for forbidden in ("commit", "revision", "ssh", "container", "remote_port", "pid"):
        assert forbidden not in resp.text.lower()


def test_stop_kgs_deployment_returns_only_public_state(monkeypatch):
    public = {
        "deployment_id": "fleet-ascend-18310-abc123",
        "instance_id": "ascend-18310",
        "machine": "ascend",
        "version": "development",
        "status": "stopped",
        "error_category": None,
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:01:00Z",
    }
    calls = []
    monkeypatch.setattr(
        "kernelgen.service.kgs_manager.stop_deployment",
        lambda deployment_id: calls.append(deployment_id) or public,
    )

    resp = _client().post(
        "/api/kgs-deployments/fleet-ascend-18310-abc123/stop"
    )

    assert resp.status_code == 200
    assert resp.json() == public
    assert calls == ["fleet-ascend-18310-abc123"]
    for forbidden in ("instance_name", "ssh", "container", "remote_port", "pid"):
        assert forbidden not in resp.text.lower()


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (KeyError("missing"), 404, "managed KGS deployment not found"),
        (ValueError("starting"), 409, "KGS deployment is not ready to stop"),
        (RuntimeError("private SSH failure"), 503, "KGS stop is temporarily unavailable"),
    ],
)
def test_stop_kgs_deployment_maps_safe_errors(monkeypatch, error, status, detail):
    def fail(_deployment_id):
        raise error

    monkeypatch.setattr("kernelgen.service.kgs_manager.stop_deployment", fail)

    resp = _client().post("/api/kgs-deployments/deployment-id/stop")

    assert resp.status_code == status
    assert resp.json()["detail"] == detail
    assert str(error) not in resp.text


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/kgs-versions", None),
        ("get", "/api/kgs-deployments", None),
        ("post", "/api/kgs-deployments/deployment-id/stop", None),
        (
            "post",
            "/api/kgs-instances",
            {"machine": "ascend", "version": "development"},
        ),
    ],
)
def test_kgs_management_endpoints_require_token(method, path, json_body):
    client = _client(ServiceConfig(token="s3cret"))

    resp = client.request(method, path, json=json_body)

    assert resp.status_code == 401


def test_index_exposes_minimal_machine_instance_version_ui():
    resp = _client().get("/")

    assert resp.status_code == 200
    html = resp.text
    assert 'id="machine"' in html
    assert 'id="device"' in html
    assert 'id="addInstanceBtn"' in html
    assert 'id="stopInstanceBtn"' in html
    assert 'id="instanceMsg"' in html
    assert 'id="addInstanceDialog"' in html
    assert 'id="kgsVersion"' in html
    assert 'id="installFlagGems" checked' in html
    assert 'install_flaggems: $("installFlagGems").checked' in html
    assert 'deployment.status === "installing_flaggems"' in html
    assert "Installing compatible FlagGems…" in html
    assert 'id="kgVersion"' in html
    assert "KernelGen Lab" in html
    assert "KGS instance" in html
    assert "color-scheme: light" in html
    assert "selectedInstanceId" in html
    assert "selectedManagedDeployment" in html
    assert 'item.status === "ready"' in html
    assert "Stop this managed KGS instance?" in html
    assert 'result.status === "stopped"' in html
    assert "Managed KGS stop is already in progress…" in html
    assert "/stop`" in html
    assert "item.address ?" in html
    assert "INSTANCE_UNAVAILABLE_LABELS" in html
    assert 'incompatible_api: "incompatible API"' in html
    assert 'scheduler_unhealthy: "scheduler unhealthy"' in html
    assert "opt.title = instanceUnavailableLabel(instance)" in html
    assert "(selected KGS unavailable)" in html
    lowered = html.lower()
    assert "commit" not in lowered
    assert "revision" not in lowered
    assert "card" not in lowered


def test_index_uses_safari_compatible_definition_combobox():
    html = _client().get("/").text

    assert "<datalist" not in html
    assert 'list="definitionList"' not in html
    assert 'role="combobox"' in html
    assert 'aria-controls="definitionListbox"' in html
    assert 'id="definitionListbox" role="listbox"' in html
    assert 'role", "option"' in html
    assert "option.textContent = name" in html
    assert "function matchingDefinitions(value)" in html
    assert "prefix.concat(contains)" in html
    assert "const DEFINITION_RENDER_LIMIT = 60" in html
    assert 'event.key === "ArrowDown"' in html
    assert 'event.key === "Enter"' in html
    assert 'event.key === "Escape"' in html
    assert "definitions available. Type to filter." in html
    assert "Definitions are temporarily unavailable." in html
    assert "?." not in html
    assert "??" not in html
    assert ".replaceChildren(" not in html
    assert ".replaceWith(" not in html


def test_index_hides_auth_controls_when_authentication_is_disabled():
    html = _client().get("/").text

    assert 'id="authControls"' in html
    assert '$("authControls").hidden = data.requires_auth === false' in html
    assert '$("authControls").hidden = false' in html
    assert 'h["X-KG-Token"] = t' in html
    assert "function restoreToken()" in html


def test_devices_endpoint_lists_presets():
    resp = _client().get("/api/devices")
    assert resp.status_code == 200
    devices = resp.json()["devices"]
    assert devices
    names = {d["name"] for d in devices}
    assert {"ascend", "nvidia"} <= names
    ascend = next(d for d in devices if d["name"] == "ascend")
    assert ascend["target_hardware"] == "Ascend910B"
    assert ascend["eval_server"].startswith("http://localhost:")


def test_definitions_endpoint_scans_default_catalog():
    resp = _client().get("/api/definitions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["catalog"] == "flaggems-adapter-definitions"
    # We don't require a specific count — just that the scan works.
    assert isinstance(body["definitions"], list)


def test_definitions_endpoint_scans_flat_catalog_layout(tmp_path, monkeypatch):
    catalog = tmp_path / "flat"
    definitions = catalog / "definitions"
    definitions.mkdir(parents=True)
    (definitions / "softmax.json").write_text("{}", encoding="utf-8")
    (definitions / "add.json").write_text("{}", encoding="utf-8")
    (definitions / "README.txt").write_text("ignore", encoding="utf-8")
    monkeypatch.setattr(
        "kernelgen.data.catalog.resolve_builtin_catalog_path",
        lambda _name: catalog,
    )
    monkeypatch.setattr(
        "kernelgen.data.catalog.load_catalog_manifest",
        lambda _name, *, path=None: {},
    )

    resp = _client().get("/api/definitions", params={"catalog": "legacy"})

    assert resp.status_code == 200
    assert resp.json()["definitions"] == ["add", "softmax"]


def test_definitions_endpoint_scans_per_operator_layout(tmp_path, monkeypatch):
    catalog = tmp_path / "native"
    for name in ("softmax", "add"):
        operator = catalog / "ops" / name
        operator.mkdir(parents=True)
        (operator / "definition.json").write_text("{}", encoding="utf-8")
    incomplete = catalog / "ops" / "incomplete"
    incomplete.mkdir(parents=True)
    monkeypatch.setattr(
        "kernelgen.data.catalog.resolve_builtin_catalog_path",
        lambda _name: catalog,
    )
    monkeypatch.setattr(
        "kernelgen.data.catalog.load_catalog_manifest",
        lambda _name, *, path=None: {"layout": "per-operator"},
    )

    resp = _client().get("/api/definitions", params={"catalog": "native"})

    assert resp.status_code == 200
    assert resp.json()["definitions"] == ["add", "softmax"]


def test_definitions_endpoint_unknown_layout_returns_empty(tmp_path, monkeypatch):
    catalog = tmp_path / "future"
    catalog.mkdir()
    monkeypatch.setattr(
        "kernelgen.data.catalog.resolve_builtin_catalog_path",
        lambda _name: catalog,
    )
    monkeypatch.setattr(
        "kernelgen.data.catalog.load_catalog_manifest",
        lambda _name, *, path=None: {"layout": "future"},
    )

    resp = _client().get("/api/definitions", params={"catalog": "future"})

    assert resp.status_code == 200
    assert resp.json()["definitions"] == []


def test_definitions_endpoint_unknown_catalog_returns_empty():
    resp = _client().get("/api/definitions", params={"catalog": "does-not-exist"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["catalog"] == "does-not-exist"
    assert body["definitions"] == []


def test_history_missing_workspace_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().get("/api/history", params={"workspace": str(tmp_path / "nope")})
    assert resp.status_code == 404


def test_activity_filters_and_projects_user_events(tmp_path, monkeypatch):
    _, workspace = _seed_run(tmp_path, monkeypatch)
    control = WorkspaceRunControl(workspace, source="test")
    control.record_event(
        "RUN_STARTED",
        stage="PREPARING",
        message="run started",
        data={"token": "secret-data-sentinel", "prompt": "secret-prompt-sentinel"},
        source="provider-token-sentinel",
    )
    control.record_event(
        "MODEL_INVOCATION_RETRYING",
        stage="MODEL_INVOCATION",
        level="WARNING",
        message="retrying model invocation",
    )
    control.record_event(
        "WORKFLOW_FAILED",
        stage="FAILED",
        level="ERROR",
        message=(
            "workflow failed at https://provider-secret.example/v1 "
            "ANTHROPIC_AUTH_TOKEN=message-token-sentinel "
            "Authorization: Bearer bearer-token-sentinel"
        ),
    )
    control.record_event("DEBUG_VISIBLE", level="DEBUG", message="debug-level")
    control.record_event("DEBUG_EVENT", visibility="DEBUG", message="debug-event")
    control.record_event("INTERNAL_EVENT", visibility="INTERNAL", message="internal")
    control.record_event("RUNTIME_LOG", message="runtime-log-sentinel")
    control.record_event(
        "MODEL_INVOCATION_FAILED",
        level="ERROR",
        message="provider-error-sentinel",
    )
    control.record_event("LONG_MESSAGE", message="x" * 2_100)

    resp = _client().get("/api/activity", params={"workspace": str(workspace)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "run_activity"
    assert body["workspace"] == str(workspace.resolve())
    assert [event["event_type"] for event in body["events"]] == [
        "RUN_STARTED",
        "MODEL_INVOCATION_RETRYING",
        "WORKFLOW_FAILED",
        "LONG_MESSAGE",
    ]
    allowed_fields = {
        "sequence",
        "recorded_at",
        "event_type",
        "scope",
        "stage",
        "level",
        "message",
    }
    assert all(set(event) == allowed_fields for event in body["events"])
    serialized = resp.text
    assert "secret-data-sentinel" not in serialized
    assert "secret-prompt-sentinel" not in serialized
    assert "provider-token-sentinel" not in serialized
    assert "provider-secret.example" not in serialized
    assert "message-token-sentinel" not in serialized
    assert "bearer-token-sentinel" not in serialized
    assert "runtime-log-sentinel" not in serialized
    assert "provider-error-sentinel" not in serialized
    long_message = body["events"][-1]["message"]
    assert len(long_message) == 2_000
    assert long_message.endswith("…")


def test_activity_initial_request_returns_recent_tail(tmp_path, monkeypatch):
    _, workspace = _seed_run(tmp_path, monkeypatch)
    control = WorkspaceRunControl(workspace)
    for index in range(1, 6):
        control.record_event("ITEM", message=str(index))

    resp = _client().get(
        "/api/activity", params={"workspace": str(workspace), "limit": 2}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [event["sequence"] for event in body["events"]] == [4, 5]
    assert body["next_sequence"] == 5
    assert body["has_more"] is False


def test_activity_incremental_pages_do_not_skip_visible_events(tmp_path, monkeypatch):
    _, workspace = _seed_run(tmp_path, monkeypatch)
    control = WorkspaceRunControl(workspace)
    for index in range(1, 6):
        control.record_event("ITEM", message=str(index))

    first = _client().get(
        "/api/activity",
        params={"workspace": str(workspace), "after_sequence": 0, "limit": 2},
    ).json()
    second = _client().get(
        "/api/activity",
        params={
            "workspace": str(workspace),
            "after_sequence": first["next_sequence"],
            "limit": 2,
        },
    ).json()
    third = _client().get(
        "/api/activity",
        params={
            "workspace": str(workspace),
            "after_sequence": second["next_sequence"],
            "limit": 2,
        },
    ).json()

    assert [event["sequence"] for event in first["events"]] == [1, 2]
    assert first["next_sequence"] == 2
    assert first["has_more"] is True
    assert [event["sequence"] for event in second["events"]] == [3, 4]
    assert second["next_sequence"] == 4
    assert second["has_more"] is True
    assert [event["sequence"] for event in third["events"]] == [5]
    assert third["next_sequence"] == 5
    assert third["has_more"] is False


def test_activity_cursor_advances_over_hidden_only_events(tmp_path, monkeypatch):
    _, workspace = _seed_run(tmp_path, monkeypatch)
    control = WorkspaceRunControl(workspace)
    control.record_event("VISIBLE")
    control.record_event("HIDDEN_DEBUG", visibility="DEBUG")
    control.record_event("HIDDEN_INTERNAL", visibility="INTERNAL")

    resp = _client().get(
        "/api/activity",
        params={"workspace": str(workspace), "after_sequence": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["events"] == []
    assert body["next_sequence"] == 3
    assert body["has_more"] is False


def test_activity_validates_workspace_and_query(tmp_path, monkeypatch):
    request, workspace = _seed_run(tmp_path, monkeypatch)
    missing = _client().get(
        "/api/activity", params={"workspace": str(tmp_path / "nope")}
    )
    assert missing.status_code == 404

    batch_workspace = tmp_path / "batch"
    save_batch_request(
        BatchRequestRecord(
            batch_id="batch-1",
            workspace=batch_workspace,
            source_path=tmp_path / "batch.yaml",
            children=[
                BatchChildRecord(
                    definition=request.definition,
                    mode=request.mode,
                    workspace=workspace,
                    run_id=request.run_id,
                )
            ],
        )
    )
    batch = _client().get(
        "/api/activity", params={"workspace": str(batch_workspace)}
    )
    assert batch.status_code == 422
    assert _client().get(
        "/api/activity",
        params={"workspace": str(workspace), "after_sequence": -1},
    ).status_code == 422
    assert _client().get(
        "/api/activity", params={"workspace": str(workspace), "limit": 101}
    ).status_code == 422


def test_activity_requires_token_when_configured(tmp_path, monkeypatch):
    _, workspace = _seed_run(tmp_path, monkeypatch)
    client = _client(ServiceConfig(token="s3cret"))
    path = "/api/activity"
    params = {"workspace": str(workspace)}
    assert client.get(path, params=params).status_code == 401
    assert (
        client.get(path, params=params, headers={"X-KG-Token": "s3cret"}).status_code
        == 200
    )


def test_cancel_seeded_run(tmp_path, monkeypatch):
    request, workspace = _seed_run(tmp_path, monkeypatch)
    resp = _client().post(
        "/api/cancel", json={"workspace": str(workspace), "reason": "test cancel"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCEL_REQUESTED"


def test_cancel_run_with_no_active_process_returns_409(tmp_path, monkeypatch):
    # cancel_run raises RuntimeError when there is no active process and no
    # server operations to forward to; the service maps that to 409 Conflict.
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resp = _client().post(
        "/api/cancel", json={"workspace": str(tmp_path / "nope"), "reason": "x"}
    )
    assert resp.status_code == 409


# --- auth ---------------------------------------------------------------------


def test_token_required_when_configured():
    client = _client(ServiceConfig(token="s3cret"))
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers={"X-KG-Token": "wrong"}).status_code == 401


def test_token_accepts_bearer_and_header(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    client = _client(ServiceConfig(token="s3cret"))
    assert client.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/runs", headers={"X-KG-Token": "s3cret"}).status_code == 200
