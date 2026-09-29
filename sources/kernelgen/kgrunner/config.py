"""Configuration loading for kgrunner."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeAgentConfig:
    bin: str = "claude"
    model: str | None = None
    base_url: str | None = None
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    max_output_tokens: int | None = None
    budget: float | None = None
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class KGRunnerConfig:
    cc: CodeAgentConfig = field(default_factory=CodeAgentConfig)
    platform_vendor: str | None = None
    gpu_ids: list[int] | None = None
    lock_dir: str = "/tmp/kgrunner_device_locks"
    output_dir: str = "./kgrunner_runs"
    poll_interval: float = 5.0


def load_config(path: Path | str) -> KGRunnerConfig:
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml required. Install: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cc_raw = raw.get("cc", {})
    cc = CodeAgentConfig(
        bin=cc_raw.get("bin", "claude"),
        model=cc_raw.get("model"),
        base_url=cc_raw.get("base_url"),
        auth_token_env=cc_raw.get("auth_token_env", "ANTHROPIC_AUTH_TOKEN"),
        max_output_tokens=cc_raw.get("max_output_tokens"),
        budget=cc_raw.get("budget"),
        extra_flags=cc_raw.get("extra_flags", []),
    )

    return KGRunnerConfig(
        cc=cc,
        platform_vendor=raw.get("platform"),
        gpu_ids=raw.get("gpu_ids"),
        lock_dir=raw.get("lock_dir", "/tmp/kgrunner_device_locks"),
        output_dir=raw.get("output_dir", "./kgrunner_runs"),
        poll_interval=raw.get("poll_interval", 5.0),
    )


def load_dotenv(env_path: Path | str | None = None) -> None:
    if env_path is None:
        env_path = Path.cwd() / ".env"
    else:
        env_path = Path(env_path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if val and val[0] in ('"', "'") and val[-1] == val[0]:
                    val = val[1:-1]
                if key:
                    os.environ[key] = val
