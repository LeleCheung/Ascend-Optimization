"""The human-readable host guide is also the single machine inventory."""
import os
from pathlib import Path
import subprocess

import pytest

from kernelgen.service import fleet_daemon as fleet


ROOT = Path(__file__).resolve().parents[1]
RECORD = "ascend-31|jump|10.0.0.31|2224|container|/workspace/kg|18326||user@secure@10.0.0.31"


def config(path):
    return fleet.FleetConfig(
        inventory=path, identity=path.parent / "unused-key",
        bastion="bastion.aiops.baai.ac.cn", jump_user="", interval=30,
        probe_timeout=5, max_parallel=3, local_port_start=18100,
        local_port_end=18200, missing_grace_scans=2,
    )


def shell_records(path):
    return subprocess.run(
        ["bash", "-c", 'source "$1"; multi_device_inventory_records', "bash",
         str(ROOT / "tests/multi_device_batch_lib.sh")],
        env={**os.environ, "MULTI_DEVICE_INVENTORY": str(path)},
        text=True, capture_output=True,
    )


def test_markdown_readers_ignore_prose_and_example_records(tmp_path, capsys):
    path = tmp_path / "hosts.md"
    path.write_text(
        "# Hosts\n" + RECORD.replace("ascend-31", "outside") + "\n"
        "```kg-hosts\n" + RECORD + "\n```\n"
        "```bash\necho example\n```\n",
    )
    targets = fleet.load_inventory(config(path))
    assert [item.name for item in targets] == ["ascend-31"]
    assert targets[0].target_hardware == "Ascend910B"
    assert capsys.readouterr().err == ""
    result = shell_records(path)
    assert result.returncode == 0
    assert result.stdout.strip() == RECORD


@pytest.mark.parametrize("text", [
    "# No inventory\n",
    "```kg-hosts\n" + RECORD,
    "```kg-hosts\n```\n```kg-hosts\n```\n",
])
def test_both_readers_reject_missing_duplicate_or_unclosed_blocks(tmp_path, text):
    path = tmp_path / "hosts.md"
    path.write_text(text)
    with pytest.raises(ValueError):
        fleet.load_inventory(config(path))
    assert shell_records(path).returncode != 0


def test_explicit_plain_inventory_still_works(tmp_path):
    path = tmp_path / "hosts.conf"
    path.write_text(RECORD + "\n")
    assert len(fleet.load_inventory(config(path))) == 1
    assert shell_records(path).stdout.strip() == RECORD


@pytest.mark.parametrize("wrapper", ["run_remote_http_proxy.sh", "run_persistent_remote_http_proxy.sh"])
def test_proxy_rejects_ambiguous_device_names(tmp_path, wrapper):
    path = tmp_path / "hosts.md"
    path.write_text("```kg-hosts\n" + RECORD + "\n" + RECORD + "\n```\n")
    identity = tmp_path / "identity"
    identity.write_text("test placeholder")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_text("#!/bin/sh\nexit 91\n")
    (fake / "python3").chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/remote_server" / wrapper),
         "--device", "ascend-31", "--inventory", str(path),
         "--identity", str(identity), "--listen-port", "19808"],
        env={**os.environ, "PATH": str(fake) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True,
    )
    assert result.returncode == 2


def test_repository_inventory_has_unique_names_and_proxy_ports(capsys):
    path = ROOT / "tests/hosts.md"
    targets = fleet.load_inventory(config(path))
    assert capsys.readouterr().err == ""
    assert len(targets) == 12
    assert len({item.name for item in targets}) == len(targets)
    assert len({item.expected_port % 100 for item in targets}) == len(targets)
    nvidia = next(item for item in targets if item.name == "nvidia")
    assert nvidia.address == "115.191.21.142"
    assert nvidia.target_hardware == "H20"
    ascend = [item for item in targets if item.name.startswith("ascend")]
    assert {item.address for item in ascend} == {"10.0.0.8", "10.0.0.9", "10.0.0.31", "10.0.0.32"}
    assert {item.container for item in ascend} == {"kernelgen-ascend-flagtree060"}
    result = shell_records(path)
    assert result.returncode == 0
    rows = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    assert len(rows) == len(targets)
    assert all(len(line.split("|")) == 9 for line in rows)
