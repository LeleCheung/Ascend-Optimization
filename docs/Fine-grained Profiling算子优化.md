# 细粒度性能分析与算子优化

1. FlagGems 仓库

    1. https://github\.com/flagos\-ai/FlagGems 看华为下的算子

2. 华为性能分析工具

    1. [Msopprof/MindStudio Insight 调优](https://jwolpxeehx.feishu.cn/wiki/Op60w1fwyidnQ7kvuLmckyOYnPe)

3. 性能上限分析工具：[Roofline 分析](https://jcnfo05ujn2x.feishu.cn/wiki/P9lkwOcR2iisKukzE0HcoLqRnNd?from=from_copylink)

    ```Python
    
    #!/usr/bin/env python3
    """Single-file, vendor-neutral Roofline analyzer.
    
    Input: one canonical JSONL file (one case x dtype per line).
    Output: aggregate tables/plots plus one folder per operator.
    """
    
    from __future__ import annotations
    
    import argparse
    import csv
    import hashlib
    import json
    import math
    import re
    from collections import Counter, defaultdict
    from dataclasses import asdict, dataclass, field
    from pathlib import Path
    from typing import Any
    
    
    SCHEMA_VERSION = "1.0"
    DTYPE_ALIASES = {
        "fp32": "float32",
        "float": "float32",
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
    }
    
    
    @dataclass
    class Point:
        dataset_id: str
        case_id: str
        op: str
        dtype: str
        status: str
        flops: float | None
        bytes: float | None
        duration_us: float | None
        compute_roof_tflops: float | None
        bandwidth_roof_gbps: float | None
        nodeid: str = ""
        vendor: str = "unknown"
        device: str = "unknown"
        implementation: str = "unknown"
        profiler: str = "unknown"
        profiler_version: str = ""
        memory_level: str = "HBM/DRAM"
        measurement_scope: str = "kernel-sum"
        flop_source: str = "unknown"
        byte_source: str = "unknown"
        time_source: str = "device"
        roof_source: str = "unknown"
        compute_path: str = "unknown"
        kernel_count: int | None = None
        kernels: list[dict[str, Any]] = field(default_factory=list)
        interval_match: bool | None = None
        duration_ratio: float | None = None
        source_path: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)
        schema_version: str = SCHEMA_VERSION
    
        @property
        def kernel_names(self) -> list[str]:
            return [
                str(item.get("name"))
                for item in self.kernels
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ]
    
    
    @dataclass
    class Result:
        point: Point
        valid: bool
        invalid_reasons: list[str]
        quality_flags: list[str]
        ai: float | None = None
        performance_tflops: float | None = None
        achieved_bandwidth_gbps: float | None = None
        knee_ai: float | None = None
        roofline_bound_tflops: float | None = None
        roofline_efficiency: float | None = None
        limiting_resource: str | None = None
        performance_band: str | None = None
    
        def row(self) -> dict[str, Any]:
            point = self.point
            return {
                "schema_version": point.schema_version,
                "dataset_id": point.dataset_id,
                "case_id": point.case_id,
                "op": point.op,
                "nodeid": point.nodeid,
                "dtype": point.dtype,
                "status": point.status,
                "vendor": point.vendor,
                "device": point.device,
                "implementation": point.implementation,
                "flops": point.flops,
                "bytes": point.bytes,
                "duration_us": point.duration_us,
                "compute_roof_tflops": point.compute_roof_tflops,
                "bandwidth_roof_gbps": point.bandwidth_roof_gbps,
                "memory_level": point.memory_level,
                "measurement_scope": point.measurement_scope,
                "profiler": point.profiler,
                "profiler_version": point.profiler_version,
                "flop_source": point.flop_source,
                "byte_source": point.byte_source,
                "time_source": point.time_source,
                "roof_source": point.roof_source,
                "compute_path": point.compute_path,
                "kernel_count": point.kernel_count,
                "kernel_names": " | ".join(point.kernel_names),
                "interval_match": point.interval_match,
                "duration_ratio": point.duration_ratio,
                "source_path": point.source_path,
                "metadata": point.metadata,
                "valid": self.valid,
                "invalid_reasons": "; ".join(self.invalid_reasons),
                "quality_flags": "; ".join(self.quality_flags),
                "arithmetic_intensity_flop_per_byte": self.ai,
                "performance_tflops": self.performance_tflops,
                "achieved_bandwidth_gbps": self.achieved_bandwidth_gbps,
                "knee_ai_flop_per_byte": self.knee_ai,
                "roofline_bound_tflops": self.roofline_bound_tflops,
                "roofline_efficiency": self.roofline_efficiency,
                "roofline_achievement_pct": (
                    self.roofline_efficiency * 100.0
                    if self.roofline_efficiency is not None
                    else None
                ),
                "limiting_resource": self.limiting_resource,
                "performance_band": self.performance_band,
            }
    
    
    def number(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
    
    
    def integer(value: Any) -> int | None:
        value = number(value)
        return int(value) if value is not None else None
    
    
    def boolean(value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
        return None
    
    
    def load_manifest(path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".json":
            raise ValueError("manifest must be one .json file")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid manifest JSON: {error.msg}") from error
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be one JSON object")
        platform = manifest.get("platform")
        if not isinstance(platform, dict):
            raise ValueError("manifest.platform must be one object")
        for name in ("vendor", "device", "implementation"):
            if not str(platform.get(name) or "").strip():
                raise ValueError(f"manifest.platform.{name} is required")
        compute_roofs = manifest.get("compute_roofs")
        bandwidth_roof = manifest.get("bandwidth_roof")
        if not isinstance(compute_roofs, dict) or not compute_roofs:
            raise ValueError("manifest.compute_roofs must be a non-empty object")
        if not isinstance(bandwidth_roof, dict) or not positive(number(bandwidth_roof.get("gbps"))):
            raise ValueError("manifest.bandwidth_roof requires positive gbps")
        normalized_roofs: dict[str, dict[str, Any]] = {}
        for dtype_name, roof in compute_roofs.items():
            dtype = DTYPE_ALIASES.get(str(dtype_name).lower(), str(dtype_name).lower())
            if not isinstance(roof, dict) or not positive(number(roof.get("tflops"))):
                raise ValueError(f"compute roof {dtype_name!r} requires positive tflops")
            if dtype in normalized_roofs:
                raise ValueError(f"duplicate normalized compute roof dtype: {dtype!r}")
            normalized_roofs[dtype] = roof
        manifest["compute_roofs"] = normalized_roofs
        return manifest
    
    
    def load_points(path: Path, manifest: dict[str, Any]) -> list[Point]:
        if path.suffix.lower() != ".jsonl":
            raise ValueError("input must be one canonical .jsonl file")
        if not path.is_file():
            raise FileNotFoundError(path)
        points: list[Point] = []
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON on line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} must be one JSON object")
                version = str(row.get("schema_version") or SCHEMA_VERSION)
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                sources = row.get("sources") if isinstance(row.get("sources"), dict) else {}
                capture = row.get("capture") if isinstance(row.get("capture"), dict) else {}
                kernels = row.get("kernels") if isinstance(row.get("kernels"), list) else []
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                dtype = str(row.get("dtype") or "unknown").replace("torch.", "").lower()
                dtype = DTYPE_ALIASES.get(dtype, dtype)
                status = str(row.get("status") or "passed")
                passed = status.lower() in {"passed", "pass", "ok", "merged", "point"}
                compute_profile = manifest["compute_roofs"].get(dtype)
                bandwidth_profile = manifest["bandwidth_roof"]
                if passed and not isinstance(compute_profile, dict):
                    raise ValueError(
                        f"line {line_number}: manifest has no compute roof for dtype {dtype!r}"
                    )
                compute_profile = compute_profile if isinstance(compute_profile, dict) else {}
                bandwidth_profile = bandwidth_profile if isinstance(bandwidth_profile, dict) else {}
                kernel_count = integer(metrics.get("kernel_count"))
                if kernel_count is None and kernels:
                    kernel_count = len(kernels)
                platform = manifest["platform"]
                profiler = manifest.get("profiler")
                profiler = profiler if isinstance(profiler, dict) else {}
                compute_source = str(compute_profile.get("source") or "unknown")
                bandwidth_source = str(bandwidth_profile.get("source") or "unknown")
                points.append(
                    Point(
                        dataset_id=str(manifest.get("dataset_id") or ""),
                        case_id=str(row.get("case_id") or ""),
                        op=str(row.get("op") or ""),
                        nodeid=str(row.get("nodeid") or ""),
                        dtype=dtype,
                        status=status,
                        vendor=str(platform.get("vendor")),
                        device=str(platform.get("device")),
                        implementation=str(platform.get("implementation")),
                        flops=number(metrics.get("flops")),
                        bytes=number(metrics.get("bytes")),
                        duration_us=number(metrics.get("duration_us")),
                        compute_roof_tflops=number(compute_profile.get("tflops")),
                        bandwidth_roof_gbps=number(bandwidth_profile.get("gbps")),
                        memory_level=str(bandwidth_profile.get("memory_level") or "HBM/DRAM"),
                        measurement_scope=str(metrics.get("scope") or "kernel-sum"),
                        profiler=str(sources.get("profiler") or profiler.get("name") or "unknown"),
                        profiler_version=str(
                            sources.get("profiler_version") or profiler.get("version") or ""
                        ),
                        flop_source=str(sources.get("flops") or "unknown"),
                        byte_source=str(sources.get("bytes") or "unknown"),
                        time_source=str(sources.get("time") or "device"),
                        roof_source=f"compute:{compute_source}; bandwidth:{bandwidth_source}",
                        compute_path=str(compute_profile.get("compute_path") or "unknown"),
                        kernel_count=kernel_count,
                        kernels=[item for item in kernels if isinstance(item, dict)],
                        interval_match=boolean(capture.get("interval_match")),
                        duration_ratio=number(capture.get("duration_ratio")),
                        source_path=str(sources.get("source_path") or path.resolve()),
                        metadata=metadata,
                        schema_version=version,
                    )
                )
        if not points:
            raise ValueError("input JSONL contains no records")
        return points
    
    
    def positive(value: float | None) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    
    
    def invalid_result(point: Point, reasons: list[str]) -> Result:
        return Result(point=point, valid=False, invalid_reasons=reasons, quality_flags=[])
    
    
    def analyze_one(point: Point, tolerance: float) -> Result:
        reasons = [
            f"missing_{name}"
            for name in ("case_id", "op", "dtype")
            if not getattr(point, name)
        ]
        passed = point.status.lower() in {"passed", "pass", "ok", "merged", "point"}
        if not passed:
            reasons.append(f"status_not_passed:{point.status}")
            return invalid_result(point, reasons)
        for name in (
            "flops",
            "bytes",
            "duration_us",
            "compute_roof_tflops",
            "bandwidth_roof_gbps",
        ):
            if not positive(getattr(point, name)):
                reasons.append(f"nonpositive_or_missing_{name}")
        if point.interval_match is False:
            reasons.append("counter_intervals_do_not_match")
        if reasons:
            return invalid_result(point, reasons)
    
        flops = float(point.flops)
        byte_count = float(point.bytes)
        duration = float(point.duration_us)
        compute = float(point.compute_roof_tflops)
        bandwidth = float(point.bandwidth_roof_gbps)
        ai = flops / byte_count
        performance = flops / duration / 1e6
        achieved_bandwidth = byte_count / duration / 1e3
        knee = compute * 1000.0 / bandwidth
        bound = min(compute, ai * bandwidth / 1000.0)
        efficiency = performance / bound
        flags: list[str] = []
        if efficiency > tolerance:
            flags.append("above_roof")
        if point.flop_source.lower().startswith("algorithmic"):
            flags.append("algorithmic_flops_not_hardware_counter")
        if point.byte_source.lower().startswith("logical"):
            flags.append("logical_bytes_not_hardware_counter")
        if point.interval_match is None:
            flags.append("counter_interval_match_unverified")
        if point.duration_ratio is not None and not 0.9 <= point.duration_ratio <= 1.1:
            flags.append("cross_capture_duration_mismatch")
        if point.flop_source.lower() in {"", "unknown"}:
            flags.append("flop_source_unknown")
        if point.byte_source.lower() in {"", "unknown"}:
            flags.append("byte_source_unknown")
        if point.time_source.lower() in {"", "unknown"}:
            flags.append("time_source_unknown")
        if point.kernel_count is not None and point.kernels and point.kernel_count != len(point.kernels):
            flags.append("kernel_count_mismatch")
        if efficiency < 0.2:
            band = "below_20pct"
        elif efficiency < 0.8:
            band = "20_to_80pct"
        else:
            band = "at_or_above_80pct"
        return Result(
            point=point,
            valid=True,
            invalid_reasons=[],
            quality_flags=flags,
            ai=ai,
            performance_tflops=performance,
            achieved_bandwidth_gbps=achieved_bandwidth,
            knee_ai=knee,
            roofline_bound_tflops=bound,
            roofline_efficiency=efficiency,
            limiting_resource="left/memory" if ai < knee else "right/compute",
            performance_band=band,
        )
    
    
    def analyze(points: list[Point], tolerance: float) -> list[Result]:
        identities = [
            (p.vendor, p.device, p.implementation, p.dtype, p.case_id) for p in points
        ]
        duplicate = {key for key, count in Counter(identities).items() if count > 1}
        results: list[Result] = []
        for point, identity in zip(points, identities):
            if identity in duplicate:
                results.append(invalid_result(point, ["duplicate_case_identity"]))
            else:
                results.append(analyze_one(point, tolerance))
        return results
    
    
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8-sig")
            return
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                if isinstance(row.get("metadata"), dict):
                    row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
                writer.writerow(row)
    
    
    def safe_name(value: str, fallback: str = "unknown", limit: int = 80) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-") or fallback
        if len(cleaned) <= limit:
            return cleaned
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        return f"{cleaned[: limit - 12]}__{digest}"
    
    
    def dtype_dir(dtype: str) -> str:
        return {
            "float32": "fp32",
            "float16": "fp16",
            "bfloat16": "bf16",
        }.get(dtype, safe_name(dtype, "unknown_dtype"))
    
    
    def roof_curve(compute: float, bandwidth: float, xs: list[float]) -> list[float]:
        return [min(compute, bandwidth * x / 1000.0) for x in xs]
    
    
    def plot_roofline(
        subset: list[Result], path: Path, title: str, annotate: bool = False
    ) -> None:
        import matplotlib.pyplot as plt
    
        first = subset[0]
        compute = float(first.point.compute_roof_tflops)
        bandwidth = float(first.point.bandwidth_roof_gbps)
        knee = compute * 1000.0 / bandwidth
        ais = [float(item.ai) for item in subset]
        performances = [float(item.performance_tflops) for item in subset]
        anchors = [*ais, knee, 1.0]
        low = 10 ** math.floor(math.log10(min(anchors)) - 0.5)
        high = 10 ** math.ceil(math.log10(max(anchors)) + 0.5)
        xs = [low * (high / low) ** (index / 399.0) for index in range(400)]
        roof = roof_curve(compute, bandwidth, xs)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.loglog(xs, roof, color="#111827", linewidth=2.2, label="100% Roofline")
        ax.loglog(xs, [0.8 * y for y in roof], "--", color="#d97706", label="80% Roofline")
        ax.loglog(xs, [0.2 * y for y in roof], ":", color="#dc2626", label="20% Roofline")
        ax.axvline(knee, color="#6b7280", linestyle="-.", label=f"Knee AI={knee:.3g}")
        ax.scatter([knee], [compute], marker="D", s=42, facecolor="white", edgecolor="#111827", zorder=5)
        colors = ["#2563eb" if item.ai < item.knee_ai else "#7c3aed" for item in subset]
        ax.scatter(ais, performances, c=colors, s=40 if len(subset) > 1 else 100, alpha=0.78, zorder=4)
        if annotate and len(subset) == 1:
            item = subset[0]
            ax.annotate(
                f"{item.point.op}\nAI={item.ai:.3g}\nPerf={item.performance_tflops:.3g} TFLOP/s\nEfficiency={item.roofline_efficiency:.1%}",
                xy=(item.ai, item.performance_tflops),
                xytext=(12, 12),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92},
            )
        ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("Performance (TFLOP/s)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
    
    
    def build_summary(results: list[Result]) -> dict[str, Any]:
        valid = [item for item in results if item.valid]
        by_dtype: dict[str, Any] = {}
        for dtype in sorted({item.point.dtype for item in results}):
            all_dtype = [item for item in results if item.point.dtype == dtype]
            usable = [item for item in all_dtype if item.valid]
            efficiencies = sorted(float(item.roofline_efficiency) for item in usable)
            median = None
            if efficiencies:
                n = len(efficiencies)
                median = efficiencies[n // 2] if n % 2 else (efficiencies[n // 2 - 1] + efficiencies[n // 2]) / 2
            by_dtype[dtype] = {
                "total": len(all_dtype),
                "valid": len(usable),
                "invalid": len(all_dtype) - len(usable),
                "knee_side": dict(Counter(item.limiting_resource for item in usable)),
                "performance_band": dict(Counter(item.performance_band for item in usable)),
                "median_roofline_efficiency": median,
                "quality_flags": dict(Counter(flag for item in usable for flag in item.quality_flags)),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "total_points": len(results),
            "valid_points": len(valid),
            "invalid_points": len(results) - len(valid),
            "invalid_reasons": dict(Counter(reason for item in results for reason in item.invalid_reasons)),
            "quality_flags": dict(Counter(flag for item in valid for flag in item.quality_flags)),
            "by_dtype": by_dtype,
        }
    
    
    def write_outputs(
        results: list[Result],
        output: Path,
        plot: bool,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {output}; use a new or empty directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "all_points.csv", [item.row() for item in results])
        valid = [item for item in results if item.valid]
        plot_paths: list[str] = []
        operator_counts = Counter((item.point.dtype, item.point.op) for item in valid)
        used_names: set[tuple[str, str]] = set()
    
        for dtype in sorted({item.point.dtype for item in results}):
            dtype_results = [item for item in valid if item.point.dtype == dtype]
            directory = output / dtype_dir(dtype)
            directory.mkdir(parents=True, exist_ok=True)
            knee_rows: list[dict[str, Any]] = []
            ranked = sorted(dtype_results, key=lambda x: (-(x.roofline_efficiency or 0), x.point.op, x.point.case_id))
            for rank, item in enumerate(ranked, 1):
                knee_rows.append({
                    "efficiency_rank": rank,
                    "case_id": item.point.case_id,
                    "op": item.point.op,
                    "nodeid": item.point.nodeid,
                    "dtype": dtype,
                    "arithmetic_intensity_flop_per_byte": item.ai,
                    "knee_ai_flop_per_byte": item.knee_ai,
                    "ai_to_knee_ratio": item.ai / item.knee_ai,
                    "knee_side": "left" if item.ai < item.knee_ai else "right",
                    "performance_tflops": item.performance_tflops,
                    "roofline_bound_tflops": item.roofline_bound_tflops,
                    "roofline_efficiency": item.roofline_efficiency,
                    "roofline_achievement_pct": item.roofline_efficiency * 100.0,
                    "achieved_bandwidth_gbps": item.achieved_bandwidth_gbps,
                    "performance_band": item.performance_band,
                    "kernel_count": item.point.kernel_count,
                    "kernel_names": " | ".join(item.point.kernel_names),
                    "quality_flags": "; ".join(item.quality_flags),
                    "source_path": item.point.source_path,
                })
            write_csv(directory / "knee_analysis.csv", knee_rows)
    
            roof_groups: dict[tuple[float, float], list[Result]] = defaultdict(list)
            for item in dtype_results:
                roof_groups[(float(item.point.compute_roof_tflops), float(item.point.bandwidth_roof_gbps))].append(item)
            for index, subset in enumerate(roof_groups.values(), 1):
                filename = "roofline.png" if len(roof_groups) == 1 else f"roofline_{index}.png"
                if plot:
                    plot_roofline(subset, directory / filename, f"Unified Roofline - {dtype}")
                    plot_paths.append(str((directory / filename).relative_to(output)))
    
            for item in dtype_results:
                op = item.point.op
                if operator_counts[(dtype, op)] == 1:
                    name = safe_name(op, "unknown_op")
                else:
                    name = safe_name(f"{op}__{item.point.case_id}", "case")
                candidate = name
                serial = 2
                while (dtype, candidate) in used_names:
                    candidate = safe_name(f"{name}__{serial}")
                    serial += 1
                used_names.add((dtype, candidate))
                case_dir = directory / "operators" / candidate
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "point.json").write_text(
                    json.dumps(item.row(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                if plot:
                    plot_roofline([item], case_dir / "roofline.png", f"Roofline - {op} ({dtype})", annotate=True)
    
        for item in (result for result in results if not result.valid):
            rejected = output / dtype_dir(item.point.dtype) / "rejected" / safe_name(
                f"{item.point.op}__{item.point.case_id}", "rejected"
            )
            rejected.mkdir(parents=True, exist_ok=True)
            (rejected / "point.json").write_text(
                json.dumps(item.row(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    
        summary = build_summary(results)
        summary["dataset"] = {
            "dataset_id": manifest.get("dataset_id", ""),
            "platform": manifest.get("platform", {}),
            "benchmark": manifest.get("benchmark", {}),
            "profiler": manifest.get("profiler", {}),
        }
        summary["plots"] = plot_paths
        summary["operator_directories"] = len(valid)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary
    
    
    def main() -> int:
        parser = argparse.ArgumentParser(description="Vendor-neutral Roofline analyzer")
        parser.add_argument("manifest", type=Path, help="shared platform and Roof manifest.json")
        parser.add_argument("points", type=Path, help="canonical points.jsonl")
        parser.add_argument("output", type=Path, help="new or empty output directory")
        parser.add_argument("--plot", action="store_true", help="generate PNG plots")
        parser.add_argument("--above-roof-tolerance", type=float, default=1.05)
        args = parser.parse_args()
        manifest = load_manifest(args.manifest.resolve())
        points = load_points(args.points.resolve(), manifest)
        results = analyze(points, args.above_roof_tolerance)
        summary = write_outputs(results, args.output.resolve(), args.plot, manifest)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    
    
    if __name__ == "__main__":
        raise SystemExit(main())
    
        
    ```

4. 分析算子瓶颈

    1. 内核级别的问题：编译出的内核数量过多，启动耗时过长。

    2. op级别：空泡

    3. 结论：一是 Triton 仍有优化空间，继续优化；二是现有实验条件下未找到有效的 Triton 优化方案。


