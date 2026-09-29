"""Entry point for running remote MCP server as a module."""

from remote_mcp.server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
