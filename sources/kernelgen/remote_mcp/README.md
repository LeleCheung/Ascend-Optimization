# KernelGen Remote MCP Server

MCP server for interacting with remote KernelGen HTTP API service.

## Overview

This MCP server provides tools to submit, monitor, and manage KernelGen optimization runs on a remote service. Unlike `mcp_server/` (workspace-internal tools for agents), this server wraps HTTP API calls for external service interaction.

## Architecture

```
Claude Code (local)
    ↓ MCP protocol (stdio)
remote_mcp/server.py
    ↓ HTTP + token auth
KernelGen HTTP Service (your-kernelgen-host:8080)
    ↓ Internal API
kernelgen.cli.api (optimization engine)
```

## Configuration

### Environment Variables

```bash
# Required: Remote service URL
export KG_API_URL="http://your-kernelgen-host:8080"

# Required: Authentication token
export KG_API_TOKEN="your-service-token-here"
```

### Claude Code Setup

Add to your `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "kernelgen-remote": {
      "command": "python3",
      "args": [
        "-m",
        "remote_mcp.server"
      ],
      "env": {
        "KG_API_URL": "http://your-kernelgen-host:8080",
        "KG_API_TOKEN": "your-service-token-here"
      },
      "cwd": "/path/to/kernelgen"
    }
  }
}
```

Or use a local configuration file `.kernelgen/remote_mcp.json`:

```json
{
  "mcpServers": {
    "kernelgen-remote": {
      "command": "python3",
      "args": ["-m", "remote_mcp.server"],
      "env": {
        "KG_API_URL": "http://your-kernelgen-host:8080",
        "KG_API_TOKEN": "your-token-here"
      }
    }
  }
}
```

## Available Tools

### kg_submit_run
Submit a new optimization run.

**Parameters:**
- `definition` (required): Kernel definition YAML/JSON or path
- `mode` (optional): Generation mode - sketch, impl, or full (default: full)
- `target_hardware` (optional): Target hardware - A100, H100, etc. (default: A100)
- `eval_server` (optional): Evaluation server endpoint
- `max_round` (optional): Maximum optimization rounds (default: 10)

**Returns:** Workspace identifier and initial status

### kg_check_status
Check status of a specific run.

**Parameters:**
- `workspace` (required): Workspace identifier from kg_submit_run

**Returns:** Current phase, progress, and results if available

### kg_list_runs
List all runs on the service.

**Returns:** Summary of active and completed runs

### kg_cancel_run
Cancel a running job.

**Parameters:**
- `workspace` (required): Workspace identifier
- `reason` (optional): Cancellation reason

**Returns:** Cancellation confirmation

### kg_get_config
Get service configuration and health.

**Returns:** Version, supported hardware, service status

## Usage Examples

### Local Testing

```bash
# Set environment
export KG_API_URL="http://localhost:8080"
export KG_API_TOKEN="your-token"

# Test MCP server
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python3 -m remote_mcp.server
```

### From Claude Code

Once configured, the tools appear automatically:

```
You: Submit a kernel optimization run for matmul on A100
Claude: [uses kg_submit_run tool]
```

## Installation

### Dependencies

```bash
pip install -r remote_mcp/requirements.txt
```

Or install individually:
```bash
pip install mcp httpx
```

Note: On dev machine (K8s pod), dependencies should be installed during service setup.

### Testing

```bash
# Unit tests
pytest remote_mcp/test_server.py

# Integration test (requires running service)
python3 remote_mcp/test_integration.py
```

## Security

- Token is transmitted via `X-KG-Token` header
- Store tokens in environment variables, never commit to git
- Use HTTPS in production (currently HTTP during development)
- Service validates tokens on every request

## Troubleshooting

### Connection Refused
- Check `KG_API_URL` points to correct host:port
- Verify service is running: `curl $KG_API_URL/api/config`
- Check network connectivity and firewall rules

### 401 Unauthorized
- Verify `KG_API_TOKEN` matches service token
- Check token is not expired or revoked

### Timeout
- Service may be under heavy load
- Check service logs for errors
- Consider increasing timeout in `httpx.AsyncClient(timeout=...)`

## Related

- HTTP Service: `service/` - FastAPI service implementation
- Service Scripts: `scripts/service/` - Start/stop/management tools  
- Workspace MCP: `mcp_server/` - Internal agent tools (different scope)
