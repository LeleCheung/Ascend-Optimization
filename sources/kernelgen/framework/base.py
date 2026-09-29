"""BaseAgent (ADR-3 #5): the Agent-as-class core.

Each agent encapsulates its input/output Pydantic models and dynamic task prompt.
Static role/tool configuration lives in KernelGen's provider-neutral
``.kernelgen/agents/*.md`` catalog. Runtime setup materializes provider files
when required. The orchestrator calls ``run(inp, runtime)``.

    run(inp, runtime)  [Runnable]:  validate input -> _execute -> validate output
    _execute  [BaseAgent]:
      1. preprocess -> prompt
      2. runtime.invoke(prompt)
      3. postprocess -> validated OutputModel
         on failure: append_repair, retry (internal robustness, max 2 repairs)

BaseAgent and Workflow share the Runnable contract, so a whole workflow is callable
and I/O-validated exactly like a single agent (composable modules).

Red line: postprocess only parses+validates the agent's PROPOSAL. Authoritative
decisions (pick_best/pick_seed/persist_kb) are the orchestrator's, called AFTER
run() returns — never inside an agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from kernelgen.framework.agent_roles import load_agent_role, render_inline_role
from kernelgen.framework.contract import append_repair, extract_json, render_contract
from kernelgen.framework.runnable import Runnable
from kernelgen.framework.runtime import Runtime


class AgentContractError(RuntimeError):
    """Raised when an agent can't produce a schema-valid output within retries."""


# Internal robustness: after the initial attempt, repair a bad agent response up
# to this many times. NOT configurable externally — it's a built-in safety net.
_MAX_REPAIRS = 2


class BaseAgent(Runnable):
    # -- static contract (subclasses set these) --------------------------
    name: str = ""
    # Legacy role template compatibility. New agents are discovered by convention
    # at .kernelgen/agents/kernel-<name>.md and keep dynamic values in preprocess().
    md_path: str = ""
    InputModel: Type[BaseModel] = BaseModel
    OutputModel: Type[BaseModel] = BaseModel
    model: str = "inherit"
    # Most agents can safely treat invalid output as a malformed final answer.
    # Stateful agents may disable this and let their supervisor inspect durable
    # lifecycle state before choosing whether to continue work or repair JSON.
    contract_repair_attempts: int = _MAX_REPAIRS
    _runtime: Optional[Any] = None        # set by bind(); used when run() has no explicit runtime

    # -- bind (from workspace path, for run_parallel) ----------------------

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "BaseAgent":
        """Construct an instance bound to a workspace runtime (for run_parallel)."""
        instance = cls()
        instance._runtime = runtime_factory(path)
        return instance

    # -- behavior (override only when needed; default = md + contract / json) --

    def preprocess(self, inp: BaseModel, runtime: Runtime) -> str:
        """Default: role + rendered input + derived output contract."""
        parts = [self._role_for(runtime)]
        parts.append("--- INPUT ---")
        parts.append(inp.model_dump_json(indent=2))
        parts.append("--- OUTPUT CONTRACT ---")
        parts.append(render_contract(self.OutputModel))
        return "\n\n".join(p for p in parts if p)

    def postprocess(self, raw: str, runtime: Runtime) -> BaseModel:
        """Default: extract JSON then validate against OutputModel."""
        return self.OutputModel.model_validate(extract_json(raw))

    # -- execution (Runnable.run validates I/O and calls this) -----------

    def _execute(self, inp: BaseModel) -> BaseModel:
        """Agent execution: build prompt -> invoke -> parse/validate + repair-retry."""
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(f"{self.name or type(self).__name__}: no runtime (pass to run() or use bind())")
        prompt = self.preprocess(inp, runtime)
        return self._run_prompt(runtime, prompt)

    def _run_prompt(
        self,
        runtime: Runtime,
        prompt: str,
        *,
        resume_session: bool = False,
    ) -> BaseModel:
        """Invoke and validate one prompt, preserving a provider session on repair."""
        base_prompt = prompt
        repair_contract = (
            "--- AUTHORITATIVE OUTPUT CONTRACT ---\n"
            + render_contract(self.OutputModel)
        )
        last_err: Optional[Exception] = None
        total_attempts = 1 + self.contract_repair_attempts
        for attempt in range(total_attempts):
            if resume_session:
                if not self._can_resume_runtime(runtime):
                    raise RuntimeError(
                        f"{self.name or type(self).__name__}: "
                        "runtime has no resumable provider session"
                    )
                raw = self._resume_runtime(runtime, prompt)
            else:
                raw = self._invoke_runtime(runtime, prompt)
            try:
                return self.postprocess(raw, runtime)
            except (ValidationError, ValueError) as e:
                last_err = e
                if attempt >= self.contract_repair_attempts:
                    continue
                if self._can_resume_runtime(runtime):
                    # The provider already has the original task and its output
                    # in context. Re-attach the exact schema because a long
                    # conversation may no longer keep the final report contract
                    # in its active context window.
                    prompt = append_repair(repair_contract, e, raw).lstrip()
                    resume_session = True
                else:
                    prompt = append_repair(
                        f"{base_prompt}\n\n{repair_contract}",
                        e,
                        raw,
                    )
        raise AgentContractError(
            f"{self.name or type(self).__name__}: no schema-valid output "
            f"after {total_attempts} attempts: {last_err}"
        )

    def continue_session(self, prompt: str, runtime: Runtime) -> BaseModel:
        """Continue the latest provider conversation with a follow-up prompt."""
        return self._run_prompt(runtime, prompt, resume_session=True)

    # -- helpers ---------------------------------------------------------

    def _invoke_runtime(self, runtime: Runtime, prompt: str) -> str:
        """Invoke through a native named agent when this provider supports it."""
        native_agent = self._native_agent_for(runtime)
        if native_agent:
            return runtime.invoke(prompt, model=self.model, agent=native_agent)
        # Preserve compatibility with third-party runtimes that implement the
        # original prompt+model contract only.
        return runtime.invoke(prompt, model=self.model)

    @staticmethod
    def _can_resume_runtime(runtime: Runtime) -> bool:
        return bool(getattr(runtime, "last_session_id", None)) and callable(
            getattr(runtime, "resume", None)
        )

    def _resume_runtime(self, runtime: Runtime, prompt: str) -> str:
        """Resume through the same native agent/provider conversation."""
        native_agent = self._native_agent_for(runtime)
        return runtime.resume(
            prompt,
            model=self.model,
            agent=native_agent,
        )

    def _native_agent_for(self, runtime: Runtime) -> Optional[str]:
        if not (
            getattr(runtime, "supports_native_agents", False)
            or getattr(runtime, "supports_agent_roles", False)
        ):
            return None
        path = self._native_definition_path()
        if path.is_file():
            return self._native_agent_name()
        if self.md_path:
            return None  # explicit legacy template with runtime placeholders
        raise RuntimeError(
            f"{self.name or type(self).__name__}: agent role definition is missing: {path}. "
            "Add the neutral role definition or declare an explicit legacy md_path."
        )

    def _native_agent_name(self) -> str:
        """Provider-neutral agent id -> Claude native file/name convention."""
        return f"kernel-{self.name.replace('_', '-')}"

    def _native_definition_path(self) -> Path:
        """Canonical provider-neutral definition in the KernelGen source tree."""
        package_root = Path(__file__).resolve().parents[1]
        return (
            package_root
            / ".kernelgen"
            / "agents"
            / f"{self._native_agent_name()}.md"
        )

    def _role_for(self, runtime: Runtime) -> str:
        """Use a provider role when available; otherwise inline the neutral role."""
        native_path = self._native_definition_path()
        if native_path.is_file() and (
            getattr(runtime, "supports_native_agents", False)
            or getattr(runtime, "supports_agent_roles", False)
        ):
            return ""
        if native_path.is_file():
            return render_inline_role(load_agent_role(native_path))
        return self._load_role()

    def _load_role(self) -> str:
        """Read role text for fallback runtimes. Missing path -> empty string.

        Relative paths resolve next to the subclass first, then from the package
        root so neutral definitions under ``.kernelgen/agents`` are reusable.
        """
        if not self.md_path:
            return ""
        p = Path(self.md_path)
        if not p.is_absolute():
            import importlib
            mod = importlib.import_module(type(self).__module__)
            agent_dir = Path(mod.__file__).resolve().parent
            candidate = agent_dir / self.md_path
            if candidate.exists():
                return self._strip_frontmatter(candidate.read_text(encoding="utf-8"))
            # Fallback: try relative to framework package root
            base = Path(__file__).resolve().parents[1]
            for fallback in [base / self.md_path, base.parents[0] / self.md_path]:
                if fallback.exists():
                    return self._strip_frontmatter(fallback.read_text(encoding="utf-8"))
        return self._strip_frontmatter(p.read_text(encoding="utf-8")) if p.exists() else ""

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Remove Claude agent YAML metadata when a role is used as fallback text."""
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return text
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "".join(lines[i + 1:]).lstrip()
        return text
