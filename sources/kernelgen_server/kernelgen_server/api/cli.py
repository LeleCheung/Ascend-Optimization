"""Command-line entrypoint for KernelGen Server."""

from __future__ import annotations

import argparse
import os

from ..operator_bundles import DEFAULT_OPERATOR_BUNDLE_MAX_BYTES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--timing",
        choices=["auto", "triton", "profiler", "walltime"],
        default="auto",
        help="auto uses the strict Ascend profiler on NPU and do_bench elsewhere",
    )
    parser.add_argument(
        "--profile-artifact-root",
        default=os.environ.get(
            "KGS_PROFILE_ARTIFACT_ROOT", "/tmp/kernelgen_server_profiles"
        ),
    )
    parser.add_argument(
        "--request-audit-root",
        default=os.environ.get("KGS_REQUEST_AUDIT_ROOT", ""),
        help=(
            "persist exact preflight/evaluate requests and execution evidence "
            "under a unique server-session directory (disabled by default)"
        ),
    )
    parser.add_argument(
        "--operator-bundle-root",
        default=os.environ.get(
            "KGS_OPERATOR_BUNDLE_ROOT", "/tmp/kernelgen_server_operator_bundles"
        ),
        help="store uploaded v6.2 native operator bundles under this directory",
    )
    parser.add_argument(
        "--operator-bundle-max-bytes",
        type=int,
        default=int(
            os.environ.get(
                "KGS_OPERATOR_BUNDLE_MAX_BYTES",
                str(DEFAULT_OPERATOR_BUNDLE_MAX_BYTES),
            )
        ),
        help="maximum archive or expanded size accepted for one operator bundle",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--enable-debug-jobs",
        dest="enable_debug_jobs",
        action="store_true",
        help=(
            "enable trusted arbitrary-command debug jobs (default); "
            "this is not a sandbox"
        ),
    )
    debug_group.add_argument(
        "--disable-debug-jobs",
        dest="enable_debug_jobs",
        action="store_false",
        help="disable the trusted debug job API",
    )
    parser.set_defaults(
        enable_debug_jobs=(
            os.environ.get("KGS_ENABLE_DEBUG_JOBS", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
    )
    parser.add_argument(
        "--debug-artifact-root",
        default=os.environ.get(
            "KGS_DEBUG_ARTIFACT_ROOT", "/tmp/kernelgen_server_debug_jobs"
        ),
    )
    parser.add_argument(
        "--debug-retention-seconds",
        type=int,
        default=int(
            os.environ.get("KGS_DEBUG_RETENTION_SECONDS", str(24 * 60 * 60))
        ),
    )
    parser.add_argument(
        "--health-probe-timeout",
        "--startup-probe-timeout",
        dest="startup_probe_timeout",
        type=float,
        default=float(
            os.environ.get(
                "KGS_HEALTH_PROBE_TIMEOUT",
                os.environ.get("KGS_STARTUP_PROBE_TIMEOUT", "30"),
            )
        ),
        help=(
            "seconds allowed for each startup or recovery device probe "
            "(default: 30; --startup-probe-timeout is a compatibility alias)"
        ),
    )
    args = parser.parse_args()
    if (
        args.enable_debug_jobs
        and args.host not in {"127.0.0.1", "localhost", "::1"}
        and os.environ.get("KGS_ALLOW_REMOTE_DEBUG") != "1"
    ):
        parser.error(
            "--enable-debug-jobs requires a loopback --host; set "
            "KGS_ALLOW_REMOTE_DEBUG=1 only behind trusted network controls"
        )
    if args.debug_retention_seconds <= 0:
        parser.error("--debug-retention-seconds must be positive")
    if args.startup_probe_timeout <= 0:
        parser.error("--startup-probe-timeout must be positive")
    if args.operator_bundle_max_bytes <= 0:
        parser.error("--operator-bundle-max-bytes must be positive")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the 'server' extra to run the HTTP service") from exc
    from .app import create_app

    uvicorn.run(
        create_app(
            args.backend,
            args.max_workers,
            args.timing,
            args.profile_artifact_root,
            enable_debug_jobs=args.enable_debug_jobs,
            debug_artifact_root=args.debug_artifact_root,
            debug_retention_seconds=args.debug_retention_seconds,
            startup_probe_timeout=args.startup_probe_timeout,
            request_audit_root=args.request_audit_root or None,
            operator_bundle_root=args.operator_bundle_root,
            operator_bundle_max_bytes=args.operator_bundle_max_bytes,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
