# Evidence layers

First verify source, contract, device, software environment, and collection scope.
Paired profiling compares actual candidate/reference device work for the same input.
Formal performance uses full-call KGS timing; profiling and simulator timing are separate.

Movement counters use totals across valid samples. Pipeline activity and latency retain
per-core distributions. Missing, negative, or non-finite metrics remain invalid/unknown.
Average movement granularity equals total bytes divided by total counts. Ratios are
observations, not causes; differing program counts require scope inspection.

Hotspot rankings use cumulative instruction time, cycles, or counts, then expand source
regions. Source maps rely on compiler positions and actual IR bound to the candidate.
Overlapping pipelines cannot be summed into kernel time. Visible idle intervals do not
by themselves establish a stall cause.

Roofline needs contract-matched work, traffic, timing, and reference bounds. Separate
same-environment microbenchmark envelopes from physical peaks, and interface traffic
from physical memory traffic. State missing inputs rather than substituting pipeline
activity for peak utilization.
