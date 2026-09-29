import csv
import json
import sqlite3
from pathlib import Path

import pytest

from kernelgen_server.profiling.cambricon.cambricon_report import (
    parse_cnperf_timeline,
    parse_mlisa,
    parse_pmu_csv,
)
from kernelgen_server.profiling.enflame.enflame_report import (
    parse_topsprof_summary,
    parse_topsprof_trace,
)
from kernelgen_server.profiling.hygon.dcu_report import (
    parse_hipprof_pmc_csv,
    parse_hipprof_timeline,
    parse_rocprof_csv,
)
from kernelgen_server.profiling.iluvatar.iluvatar_metrics import (
    MEASURED_RANGE_NAME,
    parse_ixsys_trace,
)
from kernelgen_server.profiling.metax.metax_report import parse_mctracer_trace
from kernelgen_server.profiling.mthreads.mthreads_report import analyze_mcu_output
from kernelgen_server.profiling.thead.thead import THeadACUProfiler


def test_cambricon_parsers_keep_capture_window_and_counter_units(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "setup",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 900,
                        "dur": 10,
                        "args": {},
                    },
                    {
                        "name": "cnProfilerStart",
                        "ph": "X",
                        "ts": 1000,
                        "dur": 1,
                    },
                    {
                        "name": "kernel[U1]",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 1010,
                        "dur": 1.25,
                        "pid": "MLU0 task[Async]",
                        "tid": "Q:1",
                        "args": {"dimx": 16, "visible_cluster": "8/8 (0xff)"},
                    },
                    {
                        "name": "cnProfilerStop",
                        "ph": "X",
                        "ts": 1030,
                        "dur": 0.1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_cnperf_timeline(trace)
    assert parsed["kernel_invocation_count"] == 1
    assert parsed["total_kernel_time_us"] == 1.25

    counters = tmp_path / "card0_tp_core.csv"
    counters.write_text(
        "Kernel Name,Dim,TID,Duration(ns),Visible Cluster,Device ID,"
        "alu_cycles(cycles),simd_inst_executed(counts),\n"
        '"kernel[U1]",16 * 1 * 1,42,1120,(0xff),0,100,20,\n'
        '"kernel[U1]",16 * 1 * 1,42,1120,(0xff),0,120,24,\n',
        encoding="utf-8",
    )
    pmu = parse_pmu_csv(counters)
    values = {item["name"]: item for item in pmu["kernels"][0]["counters"]}
    assert values["alu_cycles"]["unit"] == "cycles"
    assert values["alu_cycles"]["average"] == 110
    assert values["simd_inst_executed"]["sum"] == 44

    instructions = parse_mlisa(
        ".mlisa 5.0\n.visible .kernel kernel() {\n"
        "  mv.gpr.sreg %r1, %taskidx;\n  exit;\n}\n",
        binary_name="kernel.mlisa",
    )
    assert [item["opcode"] for item in instructions] == ["mv.gpr.sreg", "exit"]


def test_metax_parser_requires_runtime_markers_and_filters_setup():
    payload = {
        "traceEvents": [
            {
                "name": "setup",
                "ph": "X",
                "ts": 900_000,
                "dur": 1_000,
                "args": {"device_id": 0},
            },
            {"name": "mcProfilerStart", "ph": "X", "ts": 1_000_000, "dur": 100},
            {
                "name": "kernel",
                "ph": "X",
                "ts": 1_001_000,
                "dur": 2_000,
                "args": {
                    "device_id": 0,
                    "grid": {"x": 1, "y": 1, "z": 1},
                    "block": {"x": 256, "y": 1, "z": 1},
                    "mem": {"registers_per_thread": 4},
                },
            },
            {"name": "mcProfilerStop", "ph": "X", "ts": 1_010_000, "dur": 100},
        ]
    }
    metrics = parse_mctracer_trace(payload)
    assert metrics["kernel_invocation_count"] == 1
    assert metrics["total_kernel_time_us"] == 2.0
    assert metrics["kernels"][0]["maximum_registers_per_thread"] == 4

    payload["traceEvents"].pop()
    with pytest.raises(ValueError, match="mcProfilerStart/Stop"):
        parse_mctracer_trace(payload)


def test_enflame_summary_and_trace_are_structured(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "Type,Time(%),Time(us),Calls,Avg(us),Min(us),Max(us),Name\n"
        "GCU activities,75.00%,15.00,3,5.00,4.00,6.00,kernel\n"
        "GCU activities,25.00%,5.00,1,5.00,5.00,5.00,copy\n",
        encoding="utf-8",
    )
    parsed = parse_topsprof_summary(summary)
    assert parsed["kernel_invocation_count"] == 4
    assert parsed["total_kernel_time_us"] == 20.0
    assert parse_topsprof_trace(
        "GCU trace:\n   Start(us) Duration(us) Name\n"
        "      1.25       3.50 kernel\n"
    ) == [{"start_us": 1.25, "duration_us": 3.5, "name": "kernel"}]


def test_hygon_parsers_exclude_warmup_and_preserve_pmc(tmp_path: Path):
    timeline = tmp_path / "timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "kernel",
                        "args": {"dev_id": "0", "DurationNs": str(duration)},
                    }
                    for duration in (100_000, 200_000, 300_000)
                ]
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_hipprof_timeline(timeline, warmup=1, iterations=2)
    assert parsed["ops"][0]["measured_invocations"] == 2
    assert parsed["avg_time_us"] == 250.0

    rocprof = tmp_path / "results.csv"
    rocprof.write_text(
        "KernelName,BeginNs,EndNs\nkernel,1000,251000\n",
        encoding="utf-8",
    )
    assert parse_rocprof_csv(rocprof, warmup=0, iterations=1)["avg_time_us"] == 250.0

    pmc = tmp_path / "pmc.csv"
    pmc.write_text(
        "kernel-name,dispatch,gpu-id,time,VALUBusy,L2CacheHit\n"
        "kernel,0,0,0.000249,37.5,0.06\n",
        encoding="utf-8",
    )
    counters = parse_hipprof_pmc_csv(pmc)
    assert counters["record_count"] == 1
    assert counters["counter_names"] == ["VALUBusy", "L2CacheHit"]


def test_iluvatar_parser_aggregates_only_kernel_rows(tmp_path: Path):
    trace = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(trace)
    connection.execute(
        "CREATE TABLE cuda_kernel ("
        "ts INT, dur INT, name TEXT, deviceId INT, contextId INT, streamId INT, "
        "correlationId INT, grid TEXT, block TEXT, regs INT, srfs INT, "
        "staticSharedMemory INT, dynamicSharedMemory INT, latency INT)"
    )
    connection.execute("CREATE TABLE cuda_nvtx (ts INT, dur INT, name TEXT)")
    for index, duration in enumerate((1_000, 3_000, 2_000, 4_000), start=1):
        name = "kernel_a" if index % 2 else "kernel_b"
        connection.execute(
            "INSERT INTO cuda_kernel VALUES (?, ?, ?, 0, 1, 2, ?, ?, ?, 3, 32, 0, 64, 1)",
            (index * 10_000, duration, name, index, "(16,1,1)", "(256,1,1)"),
        )
    connection.execute(
        "INSERT INTO cuda_nvtx VALUES (0, 100000, ?)",
        (MEASURED_RANGE_NAME,),
    )
    connection.commit()
    connection.close()

    parsed = parse_ixsys_trace(trace, iterations=2)
    assert parsed["summary"]["avg_time_us"] == 5.0
    assert parsed["summary"]["kernel_invocation_count"] == 4
    assert parsed["metrics"]["measured_range_observed"] is True


def test_mthreads_and_thead_parsers_require_structured_kernel_metrics(tmp_path: Path):
    mcu_output = """==PROF== Application replay pass 0
[100] python@worker
  kernel (32, 1, 1)x(256, 1, 1), Context 0, Stream 0, Device 7
    Section: GPU Speed Of Light Throughput
    ---------------------- ----------- ------------
    Metric Name            Metric Unit Metric Value
    ---------------------- ----------- ------------
    Compute(MP) Throughput           %        45.80
    Duration                        us        22.61
    ---------------------- ----------- ------------
"""
    analysis = analyze_mcu_output(mcu_output)
    assert analysis.metrics["kernel_invocation_count"] == 1
    assert analysis.metrics["kernels"][0]["sections"][0]["metrics"][0]["value"] == 45.8

    report = tmp_path / "acu.csv"
    fields = [
        "ID",
        "Process ID",
        "Kernel Name",
        "Kernel Mangled Name",
        "Context",
        "Stream",
        "Block Size",
        "Grid Size",
        "Device",
        "Section Name",
        "Metric Name",
        "Metric Unit",
        "Metric Value",
    ]
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "ID": "7",
                "Kernel Name": "kernel",
                "Section Name": "Occupancy",
                "Metric Name": "Achieved Occupancy",
                "Metric Unit": "%",
                "Metric Value": "37.5",
            }
        )
    metrics = THeadACUProfiler()._parse_report_csv(report)
    assert metrics["kernels"][0]["sections"][0]["metrics"][0]["value"] == 37.5
