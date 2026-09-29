"""Candidate source policy owned exclusively by preflight.

Explicit violations reject execution. Uncertain receiver-method matches remain
review signals. This is a quality gate in a trusted service, not a Python sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..protocol.schema import Implementation
from .hack_detection import _canonical_name, _decorator_name, _dotted_name, _module_aliases, detect_obvious_hack

logger = logging.getLogger(__name__)
_PROTECTED = ("torch", "torch_npu", "torch_musa", "triton", "pytest")
_ENVIRON = {"os.environ", "os.environb"}
_MAPPING_WRITES = {"update", "clear", "pop", "popitem", "setdefault", "__setitem__", "__delitem__", "__ior__"}
_FILE_WRITES = {"write_text", "write_bytes", "touch", "chmod", "unlink", "rename", "replace", "mkdir", "rmdir"}
_PROCESS_MUTATIONS = {
    "os.putenv", "os.unsetenv", "os.system", "os.popen", "os.chmod", "os.remove", "os.unlink",
    "os.rename", "os.replace", "os.mkdir", "os.makedirs", "os.rmdir", "os.write",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.move", "shutil.rmtree",
    "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output", "subprocess.Popen",
}


def _environment_violations(tree: ast.AST, source_path: str) -> list[str]:
    """Recognize explicit candidate-owned process/file mutations, without executing code.

    Import and single-assignment aliases are supported. Unknown receiver methods
    are not rejected; this remains a source contract, not a Python sandbox.
    """
    aliases = {}
    for module in (*_PROTECTED, "os", "pathlib", "builtins", "io", "subprocess", "shutil"):
        aliases.update(_module_aliases(tree, module))

    def name(node):
        if isinstance(node, ast.Call):
            callee = name(node.func)
            return "pathlib.Path" if callee in {"pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath"} else ""
        if isinstance(node, ast.Attribute):
            parent = name(node.value)
            return parent + "." + node.attr if parent else ""
        if isinstance(node, ast.Subscript):
            return name(node.value)
        return _canonical_name(_dotted_name(node), aliases) or ""

    # Do not infer the type of reassigned variables or of arbitrary call results.
    bindings: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    bindings.setdefault(target.id, []).append(node.value)
    for local, values in bindings.items():
        if local not in aliases and len(values) == 1:
            resolved = name(values[0])
            if resolved:
                aliases[local] = resolved

    violations = []

    def flag(node, detail):
        violations.append(f"candidate environment mutation: {source_path}:{node.lineno} {detail}")

    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)) and name(target) in _ENVIRON:
                flag(node, name(target))
        if not isinstance(node, ast.Call):
            continue
        called = name(node.func)
        parent, _, method = called.rpartition(".")
        if called in _PROCESS_MUTATIONS or (parent in _ENVIRON and method in _MAPPING_WRITES):
            flag(node, called)
        if called.split(".")[0] in {"torch", "torch_npu", "torch_musa"} and method in {
            "empty_cache", "change_current_allocator", "_set_allocator_settings", "set_per_process_memory_fraction",
        }:
            flag(node, called)
        if parent == "pathlib.Path" and method in _FILE_WRITES:
            flag(node, called)
        if called in {"open", "builtins.open", "io.open", "pathlib.Path.open"}:
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
            offset = 0 if called == "pathlib.Path.open" else 1
            if mode is None and len(node.args) > offset:
                mode = node.args[offset]
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax+"):
                flag(node, called + " in write mode")
        if called in {"setattr", "delattr", "builtins.setattr", "builtins.delattr"} and len(node.args) >= 2:
            target, attr = name(node.args[0]), node.args[1]
            if target in _ENVIRON or (target == "os" and isinstance(attr, ast.Constant) and attr.value in {"environ", "environb"}):
                flag(node, called + " on environment")
    return violations


@dataclass(frozen=True)
class AdmissionDecision:
    rejection_reasons: tuple[str, ...]
    review_reasons: tuple[str, ...] = ()

    @property
    def is_hack(self) -> bool:
        return bool(self.rejection_reasons)

    @property
    def hack_reason(self) -> str:
        return "; ".join(self.rejection_reasons)

    @property
    def log(self) -> str:
        return "ADMISSION_REVIEW: " + "; ".join(self.review_reasons) if self.review_reasons else ""


class CandidateAdmissionError(ValueError):
    pass


@lru_cache(maxsize=1)
def admission_capability() -> dict[str, object]:
    root = Path(__file__).parent
    paths = [Path(__file__), root / "hack_detection.py", root / "metadata_policy.py"]
    blacklists = sorted((root / "blacklists").glob("*.yaml"))
    if not blacklists:
        raise RuntimeError("candidate admission policy data is missing")
    paths += blacklists
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return {"version": 1, "policy_sha256": digest.hexdigest(), "stages": ["preflight"]}


def _source_violations(implementation: Implementation, *, evaluator_kind: str) -> list[str]:
    protected_modules = _PROTECTED + (("flag_gems",) if evaluator_kind == "flaggems" else ())
    violations = []
    launches = []
    for source in implementation.sources:
        if not source.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source.content, filename=source.path)
        except SyntaxError:
            continue  # Syntax/import diagnostics belong to the ordinary loader.
        violations.extend(_environment_violations(tree, source.path))
        aliases = {}
        for module in protected_modules:
            aliases.update(_module_aliases(tree, module))

        triton_aliases = _module_aliases(tree, "triton")
        kernel_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            and any(_decorator_name(d, triton_aliases) == "triton.jit" for d in node.decorator_list)
        }

        def canonical(node):
            if isinstance(node, ast.Subscript):
                return canonical(node.value)
            return _canonical_name(_dotted_name(node), aliases) or ""

        def protected(node):
            return canonical(node).split(".")[0] in protected_modules

        def flag(node, detail):
            violations.append(f"protected candidate state: {source.path}:{node.lineno} {detail}")

        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) and protected(part):
                        flag(node, canonical(part))
                        break
            if not isinstance(node, ast.Call):
                continue
            name = canonical(node.func)
            if name in {"setattr", "delattr", "builtins.setattr", "builtins.delattr"} and node.args and protected(node.args[0]):
                flag(node, name + "(" + canonical(node.args[0]) + ", ...)")
            if name.startswith(("torch.library.", "torch._dynamo.", "torch._inductor.")) or name in {
                "torch.set_default_dtype", "torch.set_default_device", "torch.set_default_tensor_type",
                "torch.manual_seed", "torch.set_rng_state", "torch.use_deterministic_algorithms",
                "torch.backends.cuda.enable_flash_sdp", "torch.backends.cuda.enable_math_sdp",
                "torch.backends.cuda.enable_mem_efficient_sdp",
            }:
                flag(node, name)
            if evaluator_kind == "flaggems" and name in {
                "flag_gems.register", "flag_gems.runtime.register", "flag_gems.use_gems",
                "flag_gems.testing.override_registered_op", "flag_gems.testing.override_gems_op",
            }:
                flag(node, name)
            if isinstance(node.func, ast.Subscript) and (_dotted_name(node.func.value) or "").rsplit(".", 1)[-1] in kernel_names:
                grid = node.func.slice
                if isinstance(grid, ast.Lambda):
                    grid = grid.body
                dims = grid.elts if isinstance(grid, (ast.Tuple, ast.List)) else [grid]
                zero = any(isinstance(d, ast.Constant) and type(d.value) is int and d.value == 0 for d in dims)
                launches.append((source.path, node.lineno, zero))
    if launches and all(zero for _, _, zero in launches):
        violations.append("no nonzero launch: all candidate launch grids contain literal zero")
    return violations


def check_candidate_admission(implementation: Implementation, *, operator_name: str, evaluator_kind: str = "native") -> AdmissionDecision:
    admission_capability()  # Do not silently run with an unpackaged/empty policy.
    legacy = detect_obvious_hack(implementation, operator_name=operator_name)
    decision = AdmissionDecision(
        tuple(sorted(set(legacy.rejection_reasons + tuple(_source_violations(implementation, evaluator_kind=evaluator_kind))))),
        legacy.review_reasons,
    )
    if decision.review_reasons:
        logger.warning("%s", decision.log)
    return decision


def require_candidate_admission(implementation: Implementation, *, operator_name: str, evaluator_kind: str = "native") -> AdmissionDecision:
    decision = check_candidate_admission(implementation, operator_name=operator_name, evaluator_kind=evaluator_kind)
    if decision.is_hack:
        raise CandidateAdmissionError(decision.hack_reason)
    return decision
