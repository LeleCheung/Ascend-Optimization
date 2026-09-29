"""Load trusted V6 operator contracts and submitted implementations."""

from __future__ import annotations

import atexit
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Optional

from ..protocol.schema import Definition, Implementation, Parameter, SourceFile
from ..protocol.workload_call import definition_signature


class BuildError(RuntimeError):
    pass


def _cache_root() -> Path:
    configured = os.environ.get("KGS_CACHE_PATH")
    return Path(configured).expanduser() if configured else Path.home() / ".cache/kernelgen_server"


def _write_sources(root: Path, sources: Iterable[SourceFile]) -> None:
    for source in sources:
        path = root / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.content, encoding="utf-8")


def materialize_implementation(
    implementation: Implementation,
    source_root: str | Path,
) -> Path:
    """Write an implementation once at an evaluator-owned stable path."""

    root = Path(source_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_sources(root, implementation.sources)
    return root


def _verify_materialized_sources(
    source_root: Path,
    sources: Iterable[SourceFile],
) -> None:
    for source in sources:
        path = (source_root / source.path).resolve()
        if not path.is_relative_to(source_root):
            raise BuildError(f"candidate source escapes source_root: {source.path}")
        if not path.is_file():
            raise BuildError(f"materialized candidate source is missing: {source.path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BuildError(f"could not read materialized source {source.path}: {exc}") from exc
        if content != source.content:
            raise BuildError(f"materialized candidate source differs: {source.path}")


def _load_module(
    prefix: str,
    sources: list[SourceFile],
    entry_path: str,
    *,
    source_root: str | Path | None = None,
) -> ModuleType:
    key = uuid.uuid4().hex[:20]
    if source_root is None:
        parent = _cache_root() / prefix
        parent.mkdir(parents=True, exist_ok=True)
        build_root = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=parent))
        atexit.register(shutil.rmtree, build_root, ignore_errors=True)
        resolved_source_root = build_root / "source"
        resolved_source_root.mkdir(parents=True)
        _write_sources(resolved_source_root, sources)
    else:
        resolved_source_root = Path(source_root).resolve()
        if not resolved_source_root.is_dir():
            raise BuildError(f"candidate source_root is not a directory: {resolved_source_root}")
        _verify_materialized_sources(resolved_source_root, sources)

    module_name = f"_kernelgen_server_{prefix}_{key}_{os.getpid()}"
    entry = (resolved_source_root / entry_path).resolve()
    if not entry.is_relative_to(resolved_source_root) or not entry.is_file():
        raise BuildError(f"candidate entrypoint is missing: {entry_path}")
    sys.path.insert(0, str(resolved_source_root))
    spec = importlib.util.spec_from_file_location(module_name, entry)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load module from {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise BuildError(f"failed importing {entry_path}: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(resolved_source_root))
        except ValueError:
            pass
    return module


def _module_path(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        return None
    try:
        return Path(value).resolve()
    except OSError:
        return None


def _module_belongs_to(module: ModuleType, root: Path) -> bool:
    path = _module_path(module)
    if path is not None and path.is_relative_to(root):
        return True
    search_paths = getattr(module, "__path__", ())
    try:
        iterator = iter(search_paths)
    except TypeError:
        return False
    for value in iterator:
        try:
            if Path(value).resolve().is_relative_to(root):
                return True
        except (OSError, TypeError):
            continue
    return False


def _load_oracle_path(path: Path) -> ModuleType:
    """Load a trusted catalog oracle in place with its private assets visible."""

    path = path.resolve()
    operator_root = path.parent
    assets_root = operator_root / "assets"
    search_paths = [operator_root]
    if assets_root.is_dir():
        search_paths.append(assets_root)

    # A readiness retry must not silently reuse a helper imported by a previous
    # oracle instance.  Remove only modules whose source lives inside this
    # operator directory; ordinary environment modules remain untouched.
    for name, existing in list(sys.modules.items()):
        if isinstance(existing, ModuleType) and _module_belongs_to(
            existing, operator_root
        ):
            sys.modules.pop(name, None)

    module_name = f"_kernelgen_server_oracle_{uuid.uuid4().hex[:20]}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load oracle from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    inserted: list[str] = []
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for root in reversed(search_paths):
            value = str(root)
            sys.path.insert(0, value)
            inserted.append(value)
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise BuildError(f"failed importing oracle.py: {exc}") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for value in inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass
        for name, imported in list(sys.modules.items()):
            if name == module_name:
                continue
            if isinstance(imported, ModuleType) and _module_belongs_to(
                imported, operator_root
            ):
                sys.modules.pop(name, None)
    return module


def _normalize_default(value: Any, parameter: Parameter | None = None) -> Any:
    del parameter
    if value is None:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - runtime package supplies torch
        torch = None
    if torch is not None:
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, torch.device):
            return str(value)
        layouts = {
            torch.strided: "strided",
            torch.sparse_coo: "sparse_coo",
        }
        if value in layouts:
            return layouts[value]
        formats = {
            torch.contiguous_format: "contiguous_format",
            torch.preserve_format: "preserve_format",
            torch.channels_last: "channels_last",
            torch.channels_last_3d: "channels_last_3d",
        }
        if value in formats:
            return formats[value]
    if isinstance(value, tuple):
        return [_normalize_default(item) for item in value]
    return value


def validate_callable_abi(
    function: Callable[..., Any],
    definition: Definition,
    label: str,
) -> None:
    try:
        actual = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{label} has no inspectable Python signature: {exc}") from exc
    expected = definition_signature(definition)
    if len(actual.parameters) != len(expected.parameters):
        raise BuildError(
            f"{label} ABI has {len(actual.parameters)} parameters; "
            f"Definition has {len(expected.parameters)}"
        )
    for parameter, expected_parameter in zip(
        definition.parameters, actual.parameters.values(), strict=True
    ):
        wanted = expected.parameters[parameter.name]
        if expected_parameter.name != wanted.name or expected_parameter.kind != wanted.kind:
            raise BuildError(
                f"{label} ABI differs at {parameter.name!r}: expected "
                f"{wanted.kind.description} {wanted.name}, got "
                f"{expected_parameter.kind.description} {expected_parameter.name}"
            )
        wanted_required = wanted.default is inspect.Parameter.empty
        actual_required = expected_parameter.default is inspect.Parameter.empty
        if wanted_required != actual_required:
            raise BuildError(f"{label} required/default status differs for {parameter.name}")
        if not wanted_required:
            normalized = _normalize_default(expected_parameter.default, parameter)
            if type(normalized) is not type(parameter.default) or normalized != parameter.default:
                raise BuildError(
                    f"{label} default differs for {parameter.name}: "
                    f"expected {parameter.default!r}, got {normalized!r}"
                )


def _require_exact_hook(
    module: ModuleType,
    name: str,
    expected: tuple[str, ...],
) -> Optional[Callable[..., Any]]:
    function = getattr(module, name, None)
    if function is None:
        return None
    if not callable(function):
        raise BuildError(f"reference {name} must be callable")
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    if len(parameters) != len(expected) or any(
        parameter.name != wanted
        or parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter, wanted in zip(parameters, expected, strict=True)
    ):
        raise BuildError(f"reference {name} must have exact signature ({', '.join(expected)})")
    return function


def _bind_public_symbol(
    module: ModuleType,
    definition: Definition,
    label: str,
) -> Callable[..., Any]:
    run = getattr(module, "run", None)
    if not callable(run):
        raise BuildError(f"{label} run() is missing or not callable")
    validate_callable_abi(run, definition, f"{label} run")
    existing = getattr(module, definition.name, None)
    if existing is not None and existing is not run:
        raise BuildError(
            f"{label} defines conflicting public symbol {definition.name!r}"
        )
    setattr(module, definition.name, run)
    public = getattr(module, definition.name)
    validate_callable_abi(public, definition, f"{label} public symbol")
    return public


@dataclass
class OperatorAdapter:
    module: ModuleType
    reference: Optional[Callable[..., Any]]
    timing_reference: Optional[Callable[..., Any]]
    gen_inputs: Optional[Callable[..., Any]]
    validate: Optional[Callable[..., Any]]
    signature: inspect.Signature
    validate_owns_return_contract: bool = False
    reference_source: Literal["primary", "torch_fallback"] = "primary"
    has_torch_fallback: bool = False


def _valid_owns_return_contract(
    module: ModuleType,
    validator: Optional[Callable[..., Any]],
    *,
    symbol: str,
) -> bool:
    value = getattr(module, symbol, False)
    if not isinstance(value, bool):
        raise BuildError(f"reference {symbol} must be a boolean")
    if value and validator is None:
        raise BuildError(f"reference {symbol}=True requires a valid hook")
    return value


def _optional_abi_symbol(
    module: ModuleType,
    name: str,
    definition: Definition,
) -> Optional[Callable[..., Any]]:
    function = getattr(module, name, None)
    if function is None:
        return None
    if not callable(function):
        raise BuildError(f"reference {name} must be callable")
    validate_callable_abi(function, definition, f"reference {name}")
    return function


def load_operator_adapter(
    definition: Definition,
    reference_source: Literal["primary", "torch_fallback"] = "primary",
    oracle_path: str | Path | None = None,
) -> OperatorAdapter:
    if definition.reference is None:
        raise BuildError("native definition is missing trusted oracle source")
    module = (
        _load_oracle_path(Path(oracle_path))
        if oracle_path is not None
        else _load_module(
            "reference",
            [SourceFile(path="reference.py", content=definition.reference)],
            "reference.py",
        )
    )
    if definition.api_version == "v6.0":
        if reference_source != "primary":
            raise BuildError("v6.0 definitions do not support Torch fallback")
        timing_reference = _bind_public_symbol(module, definition, "reference")
        gen_inputs = _require_exact_hook(module, "gen_inputs", ("ctx", "device"))
        validator = _require_exact_hook(
            module,
            "valid",
            ("ref_outputs", "sol_outputs", "inputs", "ctx"),
        )
        reference = getattr(module, "correctness", timing_reference)
        if not callable(reference):
            raise BuildError("reference correctness must be callable")
        validate_callable_abi(reference, definition, "reference correctness")
        return OperatorAdapter(
            module=module,
            reference=reference,
            timing_reference=timing_reference,
            gen_inputs=gen_inputs,
            validate=validator,
            signature=definition_signature(definition),
        )

    torch_run = _optional_abi_symbol(module, "torch_run", definition)
    if reference_source == "torch_fallback":
        if torch_run is None:
            raise BuildError("oracle torch_run() is missing or not callable")
        reference = timing_reference = torch_run
        gen_inputs = _require_exact_hook(
            module, "torch_gen_inputs", ("ctx", "device")
        )
        validator = _require_exact_hook(
            module,
            "torch_valid",
            ("ref_outputs", "sol_outputs", "inputs", "ctx"),
        )
        validate_owns_return_contract = _valid_owns_return_contract(
            module,
            validator,
            symbol="TORCH_VALID_OWNS_RETURN_CONTRACT",
        )
    else:
        shared_reference = _optional_abi_symbol(module, "run", definition)
        correctness_override = _optional_abi_symbol(
            module, "correctness_run", definition
        )
        timing_override = _optional_abi_symbol(module, "timing_run", definition)
        reference = correctness_override or shared_reference
        timing_reference = timing_override or shared_reference
        if reference is None and timing_reference is None:
            raise BuildError(
                "v6.2 oracle must define run(), correctness_run() or timing_run()"
            )
        gen_inputs = _require_exact_hook(module, "gen_inputs", ("ctx", "device"))
        validator = _require_exact_hook(
            module,
            "valid",
            ("ref_outputs", "sol_outputs", "inputs", "ctx"),
        )
        validate_owns_return_contract = _valid_owns_return_contract(
            module,
            validator,
            symbol="VALID_OWNS_RETURN_CONTRACT",
        )
    return OperatorAdapter(
        module=module,
        reference=reference,
        timing_reference=timing_reference,
        gen_inputs=gen_inputs,
        validate=validator,
        signature=definition_signature(definition),
        validate_owns_return_contract=validate_owns_return_contract,
        reference_source=reference_source,
        has_torch_fallback=torch_run is not None,
    )


def load_implementation(
    implementation: Implementation,
    definition: Definition,
    *,
    source_root: str | Path | None = None,
) -> Callable[..., Any]:
    entry_path, symbol = implementation.entrypoint.split("::", 1)
    if symbol != "run":
        raise BuildError("V6 implementation entrypoint must be run")
    if implementation.language.value == "triton":
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise BuildError("Triton implementation requested but triton is unavailable") from exc
    module = _load_module(
        implementation.language.value,
        implementation.sources,
        entry_path,
        source_root=source_root,
    )
    return _bind_public_symbol(module, definition, "candidate")


__all__ = [
    "BuildError",
    "OperatorAdapter",
    "load_implementation",
    "load_operator_adapter",
    "materialize_implementation",
    "validate_callable_abi",
]
