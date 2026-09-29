"""Agent definition loading and IO schema validation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IOField:
    name: str
    type: str
    required: bool = True
    optional: bool = False
    desc: str = ""


@dataclass
class ResourceRequirements:
    gpu: int = 0
    workspace: bool = False
    workspace_name_key: str | None = None


@dataclass
class BackendConfig:
    type: str = "claude"
    bin: str = "claude"
    model: str | None = None
    base_url: str | None = None
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    max_output_tokens: int | None = None
    budget: float | None = None
    extra_flags: list[str] = field(default_factory=list)
    timeout: float | None = 1800
    max_retries: int = 3
    max_resumes: int = 5


@dataclass
class WorkspaceConfig:
    type: str = "git_worktree"
    repo_dir: str | None = None
    base_branch: str = "master"
    branch_prefix: str = "kgrunner"


@dataclass
class AgentDef:
    name: str
    version: str
    description: str
    prompt_template: Path
    resources: ResourceRequirements
    backend_config: BackendConfig = field(default_factory=BackendConfig)
    workspace_config: WorkspaceConfig | None = None
    inputs: list[IOField] = field(default_factory=list)
    outputs: list[IOField] = field(default_factory=list)
    skill: str | None = None

    def validate_input(self, data: dict) -> dict:
        for f in self.inputs:
            if f.required and f.name not in data:
                raise ValueError(
                    f"Agent '{self.name}' missing required input: '{f.name}'"
                )
        return data

    def validate_output(self, data: dict) -> dict:
        result = dict(data)
        for f in self.outputs:
            if f.name not in result:
                if not f.optional:
                    logger.warning(
                        "Agent '%s' output missing field '%s'", self.name, f.name
                    )
                result[f.name] = None
        return result


def load_agent(path: Path | str) -> AgentDef:
    """Load an agent definition from a directory or YAML file.

    If path is a directory, reads agent.yaml and config.yaml from it.
    If path is a file, reads it as agent.yaml and looks for config.yaml in same dir.
    """
    import yaml

    path = Path(path)

    if path.is_dir():
        agent_yaml = path / "agent.yaml"
        config_yaml = path / "config.yaml"
        base_dir = path
    else:
        agent_yaml = path
        config_yaml = path.parent / "config.yaml"
        base_dir = path.parent

    with open(agent_yaml) as f:
        raw = yaml.safe_load(f)

    resources_raw = raw.get("resources", {})
    resources = ResourceRequirements(
        gpu=resources_raw.get("gpu", 0),
        workspace=resources_raw.get("workspace", False),
        workspace_name_key=resources_raw.get("workspace_name_key"),
    )

    inputs = _parse_fields(raw.get("inputs", {}))
    outputs = _parse_fields(raw.get("outputs", {}))

    prompt_path = base_dir / raw["prompt_template"]

    # Load backend config: from config.yaml if exists, else from agent.yaml backend section
    backend_config = BackendConfig()
    workspace_config = None
    if config_yaml.exists():
        with open(config_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        backend_config = _parse_backend_config(cfg)
        if "workspace" in cfg:
            workspace_config = _parse_workspace_config(cfg)
    elif "backend" in raw:
        backend_config = _parse_backend_config(raw)
    if workspace_config is None and "workspace" in raw and isinstance(raw["workspace"], dict):
        workspace_config = _parse_workspace_config(raw)

    return AgentDef(
        name=raw["name"],
        version=raw.get("version", "0.0.0"),
        description=raw.get("description", ""),
        prompt_template=prompt_path,
        resources=resources,
        backend_config=backend_config,
        workspace_config=workspace_config,
        inputs=inputs,
        outputs=outputs,
        skill=raw.get("skill"),
    )


def load_agents_dir(directory: Path | str) -> dict[str, AgentDef]:
    """Load all agent.yaml files from a directory tree."""
    directory = Path(directory)
    agents = {}
    for yaml_path in directory.rglob("agent.yaml"):
        agent = load_agent(yaml_path.parent)
        agents[agent.name] = agent
    return agents


def _parse_fields(fields_raw: dict) -> list[IOField]:
    fields = []
    for name, spec in fields_raw.items():
        if isinstance(spec, dict):
            fields.append(IOField(
                name=name,
                type=spec.get("type", "string"),
                required=spec.get("required", True),
                optional=spec.get("optional", False),
                desc=spec.get("desc", ""),
            ))
        else:
            fields.append(IOField(name=name, type="string"))
    return fields


def _parse_backend_config(cfg: dict) -> BackendConfig:
    backend_raw = cfg.get("backend", {})
    return BackendConfig(
        type=backend_raw.get("type", "claude"),
        bin=backend_raw.get("bin", "claude"),
        model=backend_raw.get("model"),
        base_url=backend_raw.get("base_url"),
        auth_token_env=backend_raw.get("auth_token_env", "ANTHROPIC_AUTH_TOKEN"),
        max_output_tokens=backend_raw.get("max_output_tokens"),
        budget=backend_raw.get("budget"),
        extra_flags=backend_raw.get("extra_flags", []),
        timeout=cfg.get("timeout", 1800),
        max_retries=cfg.get("max_retries", 3),
        max_resumes=cfg.get("max_resumes", 5),
    )


def _parse_workspace_config(cfg: dict) -> WorkspaceConfig:
    ws_raw = cfg.get("workspace", {})
    return WorkspaceConfig(
        type=ws_raw.get("type", "git_worktree"),
        repo_dir=ws_raw.get("repo_dir"),
        base_branch=ws_raw.get("base_branch", "master"),
        branch_prefix=ws_raw.get("branch_prefix", "kgrunner"),
    )
