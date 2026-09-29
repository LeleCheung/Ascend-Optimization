#!/usr/bin/env python3
"""MCP server for remote KernelGen HTTP API calls.

This MCP server provides tools to interact with a remote KernelGen HTTP service.
Unlike mcp_server/, which provides workspace-internal tools for agents,
this server wraps HTTP API calls for external service interaction.
"""

import os
import json
from typing import Any, Optional
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent


# Configuration from environment
API_BASE_URL = os.getenv("KG_API_URL", "http://localhost:8080")
API_TOKEN = os.getenv("KG_API_TOKEN", "")

# Create server instance
server = Server("kernelgen-remote")


def _get_headers() -> dict[str, str]:
    """Get HTTP headers including authentication token."""
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["X-KG-Token"] = API_TOKEN
    return headers


async def _api_request(
    method: str,
    endpoint: str,
    *,
    json_data: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict[str, Any]:
    """Make HTTP request to KernelGen API."""
    url = f"{API_BASE_URL}{endpoint}"
    headers = _get_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        if method == "GET":
            response = await client.get(url, headers=headers, params=params)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=json_data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available KernelGen remote API tools."""
    return [
        Tool(
            name="kg_submit_run",
            description=(
                "Submit a new KernelGen optimization run to the remote service. "
                "Returns workspace path, run_id, and initial status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": "Catalog operator name / definition identifier",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Run mode (required)",
                        "enum": ["simple_opt", "kernelgen"],
                    },
                    "target_hardware": {
                        "type": "string",
                        "description": "Target hardware (default: Ascend910B)",
                    },
                    "eval_server": {
                        "type": "string",
                        "description": "Evaluation server endpoint (optional)",
                    },
                    "max_round": {
                        "type": "integer",
                        "description": "Maximum optimization rounds (default: 15)",
                    },
                    "workspace": {
                        "type": "string",
                        "description": (
                            "Optional explicit workspace path ON THE REMOTE HOST. "
                            "A timestamped default is used when omitted."
                        ),
                    },
                },
                "required": ["definition", "mode"],
            },
        ),
        Tool(
            name="kg_check_status",
            description=(
                "Check status of a specific KernelGen run or batch. "
                "Returns current state, progress, and results if available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": (
                            "Workspace path (on the remote host) from kg_submit_run"
                        ),
                    },
                },
                "required": ["workspace"],
            },
        ),
        Tool(
            name="kg_list_runs",
            description=(
                "List all KernelGen runs on the remote service. "
                "Returns summary of active and completed runs."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kg_cancel_run",
            description=(
                "Cancel a running KernelGen optimization job. "
                "Gracefully terminates the run and cleans up resources."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace path (on the remote host) of run to cancel",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for cancellation",
                    },
                },
                "required": ["workspace"],
            },
        ),
        Tool(
            name="kg_get_config",
            description=(
                "Get remote service configuration and status. "
                "Returns version, supported hardware, and service health."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kg_worker_pools",
            description=(
                "List every known worker pool (one per eval_server) with its "
                "max/used/available worker capacity and the current lease list. "
                "Use this before submitting a run to see whether a queue would "
                "form, or when diagnosing why a submitted run is still QUEUED."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kg_list_definitions",
            description=(
                "List operator names available in a built-in KernelGen Server "
                "catalog. Use this to discover valid `definition` values before "
                "calling kg_submit_run — passing an unknown operator produces "
                "an INFRASTRUCTURE_ERROR with 'unknown operator: <name>'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "catalog": {
                        "type": "string",
                        "description": (
                            "Catalog name; omit to use the service's default "
                            "(flaggems-adapter-definitions)."
                        ),
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool execution requests."""

    try:
        if name == "kg_submit_run":
            # Service contract: {definition, workspace?, options:{...}}.
            # options is validated remotely by resolve_run_options.
            options: dict[str, Any] = {"mode": arguments["mode"]}
            for key in ("target_hardware", "eval_server", "max_round"):
                if arguments.get(key) is not None:
                    options[key] = arguments[key]
            payload: dict[str, Any] = {
                "definition": arguments["definition"],
                "options": options,
            }
            if arguments.get("workspace"):
                payload["workspace"] = arguments["workspace"]
            result = await _api_request("POST", "/api/runs", json_data=payload)

        elif name == "kg_check_status":
            result = await _api_request(
                "GET", "/api/status", params={"workspace": arguments["workspace"]}
            )

        elif name == "kg_list_runs":
            result = await _api_request("GET", "/api/runs")

        elif name == "kg_cancel_run":
            result = await _api_request(
                "POST",
                "/api/cancel",
                json_data={
                    "workspace": arguments["workspace"],
                    "reason": arguments.get("reason", "cancelled via remote MCP"),
                },
            )

        elif name == "kg_get_config":
            result = await _api_request("GET", "/api/config")

        elif name == "kg_worker_pools":
            result = await _api_request("GET", "/api/worker-pools")

        elif name == "kg_list_definitions":
            params = {}
            if arguments.get("catalog"):
                params["catalog"] = arguments["catalog"]
            result = await _api_request("GET", "/api/definitions", params=params)

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False),
        )]

    except httpx.HTTPStatusError as e:
        error_msg = f"API error {e.response.status_code}: {e.response.text}"
        return [TextContent(type="text", text=error_msg)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
