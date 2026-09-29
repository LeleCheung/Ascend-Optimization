"""One optimization argument contract for the launchers, kg and YAML Batch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kernelgen.data.constants import (
    DEFAULT_CATALOG_NAME, DEFAULT_OPTIMIZATION_MODE, DEFAULT_CODER_COUNT,
    DEFAULT_EPOCH_COUNT, DEFAULT_MAX_ROUNDS,
)
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.timeout_policy import DEFAULT_EVAL_TIMEOUT_SECONDS
from kernelgen.framework.runtime import CLI_RUNTIME_NAMES
from kernelgen.knowledge.config import KnowledgeMode, KnowledgeReviewerMode


def nonnegative_int(value):
    result = int(str(value))
    if result < 0:
        raise ValueError("must be nonnegative")
    return result


def positive_int(value):
    result = nonnegative_int(value)
    if result == 0:
        raise ValueError("must be positive")
    return result


def add_optimization_arguments(parser, mode=None, *, sparse=False):
    """mode=None exposes both modes for Batch; resolved options validate one mode."""
    start = len(parser._actions)
    add = parser.add_argument
    add("--catalog-name", default=os.environ.get("KERNELGEN_CATALOG_NAME", DEFAULT_CATALOG_NAME))
    add("--catalog-path", type=Path, help="local Native or exported Gems Definition Catalog; uploaded internally to the selected KGS")
    add("--skip-review", action=argparse.BooleanOptionalAction, default=False,
        help="explicitly skip test-contract review, not target validation or code review")
    add("--eval-server", default=os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000"))
    add("--target-hardware", help="optional target constraint; defaults to the device reported by KGS")
    add("--runtime", choices=CLI_RUNTIME_NAMES, default=os.environ.get("KERNELGEN_RUNTIME", "claude"))
    add("--model", "-m")
    add("--base-url")
    add("--language", "--implementation-language", choices=[item.value for item in ImplementationLanguage], default="triton")
    add("--profile", action=argparse.BooleanOptionalAction, default=mode == "kernelgen")
    add("--early-stop-rounds", type=nonnegative_int, default=3)
    add("--min-rounds", type=nonnegative_int, default=2)
    add("--max-round", type=positive_int, default=DEFAULT_MAX_ROUNDS)
    add("--knowledge-catalog-path", type=Path)
    add("--seed-code-path", "--seed-triton-path", type=Path, help="unvalidated Triton candidate, not a timing baseline")
    add("--reference-code-path", "--reference-code", "--reference-triton-path", "--reference-triton", type=Path,
        help="read-only source in any language; may be incorrect, not a seed or timing baseline")
    add("--reference-code-prompt-path", "--reference-triton-prompt-path", type=Path)
    if mode in {None, "simple_opt"}:
        add("--warmup-ms", type=nonnegative_int, default=1000)
        add("--benchmark-ms", type=positive_int, default=100)
        add("--num-trials", type=positive_int, default=1)
        add("--eval-timeout-seconds", type=positive_int, default=DEFAULT_EVAL_TIMEOUT_SECONDS)
        add("--max-coder-sessions", type=positive_int, default=3)
        add("--dps", action=argparse.BooleanOptionalAction, default=None)
    if mode in {None, "kernelgen"}:
        add("--n-parallel", type=positive_int, default=DEFAULT_CODER_COUNT)
        add("--n-epoch", type=positive_int, default=DEFAULT_EPOCH_COUNT)
        add("--cross-epoch-knowledge", action=argparse.BooleanOptionalAction, default=True,
            help="reuse new epoch knowledge and synthesis directions; disabling permits only a read-only KB")
        add("--start-epoch", type=positive_int, default=1)
        add("--start-mode", choices=["fresh", "fork", "resume"])
        add("--finalize-epoch", type=positive_int)
        add("--timeout", type=positive_int, default=3600)
        add("--eval-atol", type=float, default=os.environ.get("KERNELGEN_EVAL_ATOL") or None)
        add("--eval-rtol", type=float, default=os.environ.get("KERNELGEN_EVAL_RTOL") or None)
        add("--eval-tolerance-mode", choices=["", "fixed", "strict"], default=os.environ.get("KERNELGEN_EVAL_TOLERANCE_MODE", ""))
        add("--eval-required-matched-ratio", type=float, default=os.environ.get("KERNELGEN_EVAL_REQUIRED_MATCHED_RATIO", "1.0"))
        add("--no-reduced-precision", action="store_true")
        add("--knowledge-mode", choices=[item.value for item in KnowledgeMode])
        add("--knowledge-derived-path", type=Path)
        add("--knowledge-reviewer-mode", choices=[item.value for item in KnowledgeReviewerMode], default=os.environ.get("KERNELGEN_KNOWLEDGE_REVIEWER_MODE", "off"))
        add("--knowledge-run-archive-path", type=Path)
        add("--knowledge-run-id", default="")
    if sparse:
        for action in parser._actions[start:]:
            action.default = None


def optimization_parser(mode=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_optimization_arguments(parser, mode)
    return parser


def validate_options(values, *, mode=None, base=None):
    """Validate sparse YAML/CLI values with the very same argparse contract."""
    actions = {action.dest: action for action in optimization_parser(mode)._actions}
    aliases = {option[2:].replace("-", "_"): action.dest
               for action in actions.values() for option in action.option_strings
               if option.startswith("--") and action.nargs != 0}
    result = {}
    for name, value in values.items():
        name = aliases.get(name, name)
        if name == "mode":
            if value not in {"simple_opt", "kernelgen"}:
                raise ValueError("mode must be simple_opt or kernelgen")
            result[name] = value
            continue
        if name not in actions:
            raise ValueError(f"unsupported option {name!r} for mode={mode or 'any'}")
        if value is None:
            continue
        action = actions[name]
        try:
            if action.nargs == 0:
                if not isinstance(value, bool):
                    raise ValueError("must be a boolean")
            elif action.type is not None:
                value = action.type(value)
            elif not isinstance(value, str):
                raise ValueError("must be a string")
            if action.choices is not None and value not in action.choices:
                raise ValueError(f"must be one of {list(action.choices)}")
            if isinstance(value, Path):
                value = value.expanduser()
                value = ((base or Path.cwd()) / value).resolve()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name}: {exc}") from exc
        if name in result and result[name] != value:
            raise ValueError(f"conflicting values for {name}")
        result[name] = value
    return result


def resolve_run_options(*layers):
    explicit = {}
    for layer in layers:
        explicit.update(layer)
    mode = explicit.get("mode") or DEFAULT_OPTIMIZATION_MODE
    if mode not in {"simple_opt", "kernelgen"}:
        raise ValueError("mode must be simple_opt or kernelgen")
    parser = optimization_parser(mode)
    defaults = vars(parser.parse_args([]))
    if explicit.get("catalog_path") is not None:
        if explicit.get("catalog_name") is not None:
            raise ValueError("select exactly one of --catalog-name or --catalog-path")
        defaults["catalog_name"] = None
    values = validate_options({**defaults, **explicit}, mode=mode)
    # Keep optional keys available to launchers and request construction.
    return {**defaults, **values, "mode": mode}


def optimization_argv(values):
    """Serialize the resolved contract; managed options have no second parse path."""
    result = []
    for action in optimization_parser(values["mode"])._actions:
        value = values.get(action.dest)
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(action.option_strings[0 if value else 1])
        elif action.nargs == 0:
            if value:
                result.append(action.option_strings[0])
        else:
            result.extend([action.option_strings[0], str(value)])
    return result


def launcher_parser(mode, *, sparse=False):
    parser = argparse.ArgumentParser(description=f"{mode}: optimize a catalog operator", allow_abbrev=False)
    parser.add_argument("--definition", "--definition-name", "-d", "-n", required=True)
    parser.add_argument("--workspace", "-w", type=Path)
    add_optimization_arguments(parser, mode, sparse=sparse)
    # Direct launchers remain usable independently; kg never persists credentials
    # or permits deleting an existing campaign with --clean.
    parser.add_argument("--auth-token")
    parser.add_argument("--clean", action="store_true")
    return parser
