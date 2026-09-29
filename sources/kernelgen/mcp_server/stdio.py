"""Asyncio-native MCP stdio transport for Linux agent workspaces.

The upstream MCP transport wraps blocking standard streams with AnyIO worker
threads. Some managed execution environments do not reliably deliver the
worker-thread completion callback to the asyncio loop, which stalls the MCP
initialize handshake. KernelGen keeps the official MCP message/session/server
implementation and replaces only standard-input reading with asyncio's native
pipe transport. Standard-output writes remain serialized in the event-loop
thread, where Claude Code continuously drains the pipe.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.shared.message import SessionMessage


@asynccontextmanager
async def asyncio_stdio_server():
    """Yield official MCP memory streams backed by asyncio's stdin pipe."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    read_transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    read_sender: MemoryObjectSendStream[SessionMessage | Exception]
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_receiver: MemoryObjectReceiveStream[SessionMessage]
    read_sender, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_receiver = anyio.create_memory_object_stream(0)

    async def stdin_reader() -> None:
        try:
            async with read_sender:
                while line := await reader.readline():
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await read_sender.send(exc)
                        continue
                    await read_sender.send(SessionMessage(message))
        except (anyio.ClosedResourceError, asyncio.CancelledError):
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_receiver:
                async for session_message in write_receiver:
                    encoded = session_message.message.model_dump_json(
                        by_alias=True,
                        exclude_none=True,
                    ).encode("utf-8") + b"\n"
                    sys.stdout.buffer.write(encoded)
                    sys.stdout.buffer.flush()
        except (anyio.ClosedResourceError, asyncio.CancelledError):
            await anyio.lowlevel.checkpoint()

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(stdin_reader)
            task_group.start_soon(stdout_writer)
            yield read_stream, write_stream
    finally:
        read_transport.close()


class KernelGenFastMCP(FastMCP):
    """FastMCP using the asyncio-native stdio transport on Unix."""

    async def run_stdio_async(self) -> None:
        async with asyncio_stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )
