"""Runtime setup for OperatorOptimize, shared by kg run and legacy examples."""

from pathlib import Path

from kernelgen.framework import copy_claude_directory, copy_mcp_configuration
from kernelgen.framework.runtime import create_cli_runtime, resolve_cli_runtime_options, materialize_claude_runtime_config
from kernelgen.data.timeout_policy import DEFAULT_EVAL_TIMEOUT_SECONDS, TimeoutPolicy
from kernelgen.workflows.optimization import OperatorOptimizeWorkflow


def run_operator_optimization(inp, *, workspace, runtime="claude",
                         model=None, base_url=None, auth_token=None, resume=False):
    """Python/CLI entry: pass a validated request directly, never rebuild argv."""
    workspace = Path(workspace).expanduser().resolve()
    inp = {**inp, "resume": resume}
    options = resolve_cli_runtime_options(runtime, model=model, base_url=base_url, auth_token=auth_token)
    policy = TimeoutPolicy(inp.get("optimization", {}).get("eval_timeout_seconds", DEFAULT_EVAL_TIMEOUT_SECONDS))
    source = Path(__file__).resolve().parents[1]

    def make_runtime(path):
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        optimization = root.is_relative_to(workspace / "stages/optimize/work")
        kwargs = {}
        copy_claude_directory(source / ".claude", root / ".claude", include_skills=optimization)
        if optimization:
            copy_mcp_configuration(source / ".kernelgen/mcp.json", root / ".mcp.json",
                                   tool_timeout_seconds=policy.coder_idle_timeout_seconds)
        elif runtime == "claude":
            kwargs["allowed_tools"] = "Read"
        else:
            kwargs["sandbox_mode"] = "read-only"
        config = materialize_claude_runtime_config(root, options["model"]) if runtime == "claude" else None
        timeout = inp.get("optimization", {}).get("timeout", policy.coder_hard_timeout_seconds)
        return create_cli_runtime(runtime, workspace=root, **options, claude_config_dir=config,
                                  timeout=timeout,
                                  idle_timeout=policy.coder_idle_timeout_seconds, **kwargs)

    result = OperatorOptimizeWorkflow(cwd=workspace, runtime_factory=make_runtime).run(inp)
    print(result.model_dump_json(indent=2))
    return 0 if result.state == "SUCCEEDED" else 130 if result.state == "CANCELLED" else 1


def run_optimization_example(mode, argv=None):
    """The two optimization examples use the same options and Workflow as kg."""
    import uuid
    from kernelgen.framework.catalog_options import catalog_input
    from kernelgen.framework.local_state import state_home
    from kernelgen.framework.run_options import launcher_parser, resolve_run_options

    parser = launcher_parser(mode, sparse=True)
    args = parser.parse_args(argv)
    if args.clean:
        parser.error("Catalog workflows preserve prior runs; use a new --workspace, not --clean")
    explicit = {key: value for key, value in vars(args).items()
                if key not in {"definition", "workspace", "auth_token", "clean"} and value is not None}
    values = resolve_run_options({"mode": mode, **explicit})
    inp = catalog_input(values, args.definition)
    workspace = args.workspace or state_home() / "runs" / f"{args.definition}-{uuid.uuid4().hex[:8]}"
    if not (workspace / ".kernelgen/operator-lifecycle.json").exists() and (
        (workspace / ".ledger.json").exists() or any(workspace.glob("*R/agent*/.ledger.json"))
    ):
        parser.error("legacy Python workspace: resume with legacy_main; do not migrate its ledger layout")
    return run_operator_optimization(inp.model_dump(mode="json"), workspace=workspace,
                                runtime=values["runtime"], model=values.get("model"),
                                base_url=values.get("base_url"), auth_token=args.auth_token,
                                resume=values.get("start_mode") == "resume" or values.get("start_epoch", 1) > 1)
