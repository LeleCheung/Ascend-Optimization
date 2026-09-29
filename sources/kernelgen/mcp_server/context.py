"""Compatibility imports for the workspace-bound tool context.

New code should import these definitions from :mod:`kernelgen.data.tool_context`.
"""

from kernelgen.data.tool_context import (
    ToolContext,
    load_tool_context,
    resolve_workspace_file,
    workspace_from_env,
)

__all__ = [
    "ToolContext",
    "load_tool_context",
    "resolve_workspace_file",
    "workspace_from_env",
]
