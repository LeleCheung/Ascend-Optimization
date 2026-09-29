"""Code agent subprocess management."""

from .base import CodeAgentBackend, CodeAgentConfig, CodeAgentProcess, default_output_parser
from .claude import ClaudeBackend, spawn_agent, resume_agent, kill_agent

__all__ = [
    "CodeAgentBackend",
    "CodeAgentConfig",
    "CodeAgentProcess",
    "ClaudeBackend",
    "default_output_parser",
    "spawn_agent",
    "resume_agent",
    "kill_agent",
]
