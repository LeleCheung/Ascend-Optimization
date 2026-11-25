Load skills/hgkda/SKILL.md. Establish a strict Triton start for CANNBench Level4 GQA.
Read contract.json, task-description.md, and official golden.py. Read/write only this
workspace; no historical candidates, other experiments, or other agents.

candidate.py is empty. Implement run(query,key,value,scaleValue=-1.0,is_causal=False),
returning BSND with query shape/dtype. Support all four official cases using actual
shape/stride/parameters, never case IDs, values, seeds, or cached answers. Map query
heads to shared KV heads without materializing repeats. Causal masks align bottom-right.

All device work uses Triton; Python only allocates, handles metadata, and launches.
torch.empty is allowed; PyTorch matmul/softmax/indexing/native attention cannot replace
candidate work. Establish accuracy and valid UB before performance. Keep MERE/MARE
thresholds; extra allclose is separate from official accuracy.

Choose tiled/online-softmax or equivalent structure. Assess low-precision probability
and reduction relative error near zero; use adequate accumulation precision and explain it.

Edit only candidate.py and diagnosis.json. Before implementation register source SHA
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, official semantics/contract,
hypothesis, structure, expectations, and validation through bridge decision --record
diagnosis.json. Evidence paths may be run-relative, e.g. workspace/task-description.md;
category=knowledge, role=background. Do not invent profiling/simulation.

Generate the start and hand off; hardware evaluation follows host review. Do not run
preflight/environment probes. Briefly explain numerical/layout/UB choices in English.
Repair actual compilation/accuracy feedback later. Do not copy golden device math.
