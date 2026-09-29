"""KernelGen framework: the public API surface."""

from .agent_roles import (
    AgentRole,
    canonical_agent_roles_dir,
    load_agent_role,
    load_agent_roles,
    materialize_agent_role,
    materialize_agent_roles,
    render_inline_role,
)
from .agent_skills import (
    AGENT_SKILLS_DIRECTORY,
    AgentSkill,
    canonical_agent_skills_dir,
    load_agent_skill,
    load_agent_skills,
    materialize_agent_skills,
    remove_materialized_agent_skills,
)
from .base import AgentContractError, BaseAgent
from .contract import append_repair, extract_json, render_contract
from .mcp_config import (
    MCP_CONFIGURATION_PATH,
    MCPServerConfiguration,
    canonical_mcp_configuration_path,
    load_kernelgen_mcp_configuration,
    load_mcp_configuration,
    materialize_mcp_configuration,
)
from .models import DefinitionModel
from .parallel import (
    Directory,
    IsolatedDirectory,
    ParallelExecutionError,
    ParallelTaskFailure,
    Workspace,
    copy_claude_directory,
    copy_mcp_configuration,
    run_parallel,
)
from .runnable import Runnable
from .run_control import (
    CancellationState,
    CancellationController,
    CancellationToken,
    EventSink,
    EventStream,
    ProgressRecord,
    ProgressStore,
    RunCancelled,
    RunControl,
    RunEvent,
    RunProgressSnapshot,
    RunState,
    WorkspaceRunControl,
    cooperative_cancel_result,
)
from .runtime import (
    CLIRuntime,
    ClaudeCodeRuntime,
    ClaudeRuntime,
    CLI_RUNTIME_NAMES,
    CodexRuntime,
    FakeRuntime,
    Runtime,
    create_cli_runtime,
    materialize_claude_runtime_config,
    resolve_cli_runtime_options,
)
from .workflow import Workflow

__all__ = [
    "Runnable", "BaseAgent", "Workflow",
    "AgentContractError",
    "AgentRole", "canonical_agent_roles_dir", "load_agent_role",
    "load_agent_roles", "materialize_agent_role", "materialize_agent_roles",
    "render_inline_role",
    "AGENT_SKILLS_DIRECTORY", "AgentSkill", "canonical_agent_skills_dir",
    "load_agent_skill", "load_agent_skills", "materialize_agent_skills",
    "remove_materialized_agent_skills",
    "MCP_CONFIGURATION_PATH", "MCPServerConfiguration",
    "canonical_mcp_configuration_path", "load_mcp_configuration",
    "load_kernelgen_mcp_configuration", "materialize_mcp_configuration",
    "Runtime", "FakeRuntime", "CLIRuntime", "ClaudeRuntime", "ClaudeCodeRuntime",
    "CodexRuntime",
    "CLI_RUNTIME_NAMES", "create_cli_runtime", "resolve_cli_runtime_options",
    "materialize_claude_runtime_config",
    "Workspace", "Directory", "IsolatedDirectory", "copy_claude_directory",
    "copy_mcp_configuration",
    "ParallelTaskFailure", "ParallelExecutionError", "run_parallel",
    "CancellationState", "CancellationController", "CancellationToken",
    "EventSink", "EventStream",
    "ProgressRecord", "ProgressStore", "RunCancelled", "RunControl", "RunEvent",
    "RunProgressSnapshot", "RunState", "WorkspaceRunControl",
    "cooperative_cancel_result",
    "DefinitionModel",
    "render_contract", "extract_json", "append_repair",
]
