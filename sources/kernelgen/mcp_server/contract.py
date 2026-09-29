"""Stable public names shared by MCP registration and configuration checks."""

MCP_SERVER_NAME = "kernelgen"
MCP_TOOL_NAMES = frozenset(
    {
        "get_server_status",
        "preflight_kernel",
        "submit_debug_job",
        "eval_round",
        "request_retest",
        "eval_only",
        "finalize_round",
        "get_profile_context",
        "profile_workloads",
        "record_profile_analysis",
        "query_knowledge",
        "get_knowledge",
        "query_sources",
        "get_source",
        "search_rounds",
    }
)
MCP_CLAUDE_TOOL_NAMES = frozenset(
    f"mcp__{MCP_SERVER_NAME}__{name}" for name in MCP_TOOL_NAMES
)
