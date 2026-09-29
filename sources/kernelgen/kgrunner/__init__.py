"""KernelGenRunner — Agent orchestration framework."""

from .agent import AgentDef, BackendConfig, load_agent, load_agents_dir
from .codeagent import CodeAgentBackend, CodeAgentConfig, CodeAgentProcess, ClaudeBackend, default_output_parser
from .config import KGRunnerConfig, load_config, load_dotenv
from .device import GPUPool
from .map import run_parallel
from .platform import PlatformInfo, PLATFORM_REGISTRY, detect_platform, make_device_env
from .progress import TaskTracker, TaskStatus
from .run import run
from .template import render_template
from .workspace import Workspace, GitWorktree, TempDir, Directory

__all__ = [
    "run",
    "run_parallel",
    "GPUPool",
    "Workspace",
    "GitWorktree",
    "TempDir",
    "Directory",
    "AgentDef",
    "BackendConfig",
    "load_agent",
    "load_agents_dir",
    "CodeAgentBackend",
    "CodeAgentConfig",
    "CodeAgentProcess",
    "ClaudeBackend",
    "default_output_parser",
    "KGRunnerConfig",
    "load_config",
    "load_dotenv",
    "PlatformInfo",
    "PLATFORM_REGISTRY",
    "detect_platform",
    "make_device_env",
    "TaskTracker",
    "TaskStatus",
    "render_template",
]
